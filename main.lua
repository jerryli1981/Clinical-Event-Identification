--[[
Main Driver for Crepe
By Xiang Zhang @ New York University
]]

-- Necessary functionalities
require("nn")


-- Local requires
require("data")
require("model")
require("train")
require("test")

require('lfs')

-- Configurations
dofile("config.lua")

-- Prepare random number generator
math.randomseed(os.time())
torch.manualSeed(os.time())

-- Create namespaces
main = {}

-- The main program
function main.main()

   opt = main.argparse()
   -- Setting the device
   if opt.device > 0 then
      require("cutorch")
      require("cunn")
      cutorch.setDevice(opt.device)
      print("Device set to ".. opt.device)
      config.main.type = "torch.CudaTensor"
   else
      config.main.type = "torch.DoubleTensor"
   end

   if opt.debug > 0 then
      dbg = require("debugger")
   end

   main.clock = {}
   main.clock.log = 0
   

   if opt.test > 0 then
      main.test()
   else
      main.new()
      main.run()
   end
 
end

-- Parse arguments
function main.argparse()
   local cmd = torch.CmdLine()

   -- Options
   cmd:option("-resume",0,"Resumption point in epoch. 0 means not resumption.")
   cmd:option("-test",0,"test. 0 means not test.")
   cmd:option("-debug",0,"debug. 0 means not debug.")
   cmd:option("-device",0,"device. 0 means cpu.")
   cmd:text()

   -- Parse the option
   local opt = cmd:parse(arg or {})

   -- Resumption operation
   if opt.resume > 0 then
      -- Find the main resumption file
      local files = main.findFiles(paths.concat(config.main.save,"main_"..tostring(opt.resume).."_*.t7b"))
      if #files ~= 1 then
    error("Found "..tostring(#files).." main resumption point.")
      end
      config.main.resume = files[1]
      print("Using main resumption point "..config.main.resume)
      -- Find the model resumption file
      local files = main.findFiles(paths.concat(config.main.save,"sequential_"..tostring(opt.resume).."_*.t7b"))
      if #files ~= 1 then
    error("Found "..tostring(#files).." model resumption point.")
      end
      config.model.file = files[1]
      print("Using model resumption point "..config.model.file)
      -- Resume the training epoch
      config.train.epoch = tonumber(opt.resume) + 1
      print("Next training epoch resumed to "..config.train.epoch)
      -- Don't do randomize
      if config.main.randomize then
    config.main.randomize = nil
    print("Disabled randomization for resumption")
      end
   end

   return opt
end

-- Train a new experiment
function main.new()
   -- Load the data
   print("Loading datasets...")
   main.train_data = Data(config.train_data)
   main.val_data = Data(config.val_data)
   
   -- Load the model
   print("Loading the model...")
   main.model = Model(config.model)
   if config.main.randomize then
      main.model:randomize(config.main.randomize)
      print("Model randomized.")
   end
   main.model:type(config.main.type)
   print("Current model type: "..main.model:type())
   collectgarbage()

   -- Initiate the trainer
   print("Loading the trainer...")
   main.train = Train(main.train_data, main.model, config.loss(), config.train)

   -- Initiate the tester
   print("Loading the tester...")
   main.test_val = Test(main.val_data, main.model, config.loss(), config.test)

   -- The record structure
   main.record = {}

   collectgarbage()
end

-- Start the training
function main.run()
   --Run for this number of era
   local best_acc_score = -1.0
   for i = 1,config.main.eras do

      if config.main.dropout then
	     print("Enabling dropouts")
	     main.model:enableDropouts()
      else
	     print("Disabling dropouts")
	     main.model:disableDropouts()
      end
      print("Training for era "..i)
      main.train:run(config.main.epoches, main.trainlog)

      if config.main.validate == true then
	     print("Disabling dropouts")
        main.model:disableDropouts()
	     print("Testing on develop data for era "..i)
	     main.test_val:run(main.testlog)
      end

      acc_score = 1 - main.test_val.e
      print("Validate accuracy is: " .. string.format("%.2e",acc_score))
      if acc_score > best_acc_score then
         main.save()
      end
      collectgarbage()
   end
end

function main.test()
   print("Begin Testing ....")
   -- Load the model
   print("Loading the model...")
   if config.model.file == nil then
      error("need set model file")
   end

   main.model = Model(config.model)
   main.model:type(config.main.type)

   print("Disabling dropouts")
   main.model:disableDropouts()

   main.test_data = Data(config.test_data)

   main.test = Test(main.test_data, main.model, config.loss(), config.test)
   main.test:run(main.testlog)
   preds = main.test.predLabels

   local preds_out = torch.DiskFile(paths.cwd() .. "/span_decisions.txt", 'w')
   for i = 1, preds:size(1) do
    preds_out:writeString(preds[i] .. "\n")
   end
   print("predicted events is: " .. preds:size(1)) -- 96326
   preds_out:close()

   os.execute("python makeEvaluation.py")

   --[[

   fs = paths.iterdirs(paths.cwd() .. "/annotation/coloncancer/Test")
   idx = 0

   allcase = {}
   for fn in fs do
      table.insert(allcase,fn)
   end
   table.sort(allcase)

   for i=1, #allcase do
      fn = allcase[i]
      f_dir = paths.cwd() .. "/annotation/coloncancer/Test/" .. fn

      output_dir = paths.cwd() .. "/output"
      if lfs.attributes(output_dir) == nil then
         lfs.mkdir(output_dir)
      end

      out_dir = output_dir .. "/" .. fn
      if lfs.attributes(out_dir) == nil then
         lfs.mkdir(out_dir)
      end

      xmls = paths.dir(f_dir)

      local xml
      for i=1, #xmls do
         tmpxml = xmls[i]
         if string.find(tmpxml, "Temporal") then
            xml = tmpxml
         end
      end

      local x_n_out = torch.DiskFile(out_dir .. "/" .. xml, 'w')
      local file = io.open(f_dir .. "/" .. xml, 'r')

      while true do
       local line = file:read()
       if line == nil then break end

       if string.find(line, "<Type>N/A</Type>") or 
         string.find(line, "<Type>ASPECTUAL</Type>")  or string.find(line, "<Type>EVIDENTIAL</Type>") then

         idx = idx + 1

         if preds[idx] == 1 then
            line = "<Type>" .. "N/A" .. "</Type>"
         elseif preds[idx] == 2 then
            line = "<Type>" .. "ASPECTUAL" .. "</Type>"
         elseif preds[idx] == 3 then
            line = "<Type>" .. "EVIDENTIAL" .. "</Type>"
         else 
            error("Wrong label")         
         end

       end

       x_n_out:writeString(line .. "\n")

      end    

      file:close()
      x_n_out:close()

   end

   if idx ~= preds:size(1) then
      error("Label numer mismatch")
   end

   os.execute("python -m anafora.evaluate -r annotation/coloncancer/Test/ -p output")
   --]]

end

function main.save()
   -- Record necessary configurations
   config.train.epoch = main.train.epoch

   if lfs.attributes(config.main.save) == nil then
         lfs.mkdir(config.main.save)
   end

   -- Make the save
   local time = os.time()
   torch.save(paths.concat(config.main.save,"main_"..(main.train.epoch-1).."_"..time..".t7b"),
         {config = config, record = main.record, momentum = main.train.old_grads:double()})
   torch.save(paths.concat(config.main.save,"sequential_"..(main.train.epoch-1).."_"..time..".t7b"),
         main.model:clearSequential(main.model:makeCleanSequential(main.model.sequential)))

   collectgarbage()
end

-- The training logging function
function main.trainlog(train)
   if config.main.collectgarbage and math.fmod(train.epoch-1,config.main.collectgarbage) == 0 then
      print("Collecting garbage at epoch = "..(train.epoch-1))
      collectgarbage()
   end

   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      local msg = ""

	     msg = msg.."epo: "..(train.epoch-1)..
	    ", rat: "..string.format("%.2e",train.rate)..
	    ", err: "..string.format("%.2e",train.error)..
	    ", obj: "..string.format("%.2e",train.objective)

      print(msg)
   
      main.clock.log = os.time()
   end
end

function main.testlog(test)
   if config.main.collectgarbage and math.fmod(test.n,config.train_data.batch_size*config.main.collectgarbage) == 0 then
      print("Collecting garbage at n = "..test.n)
      collectgarbage()
   end
   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      print("n: "..test.n..
	       ", e: "..string.format("%.2e",test.e)..
	       ", l: "..string.format("%.2e",test.l)..
	       ", err: "..string.format("%.2e",test.err)..
	       ", obj: "..string.format("%.2e",test.objective))
      main.clock.log = os.time()
   end
end

-- Utility function: find files with the specific 'ls' pattern
function main.findFiles(pattern)
   require("sys")
   local cmd = "ls "..pattern
   local str = sys.execute(cmd)
   local files = {}
   for file in str:gmatch("[^\n]+") do
      files[#files+1] = file
   end
   return files
end

-- Execute the main program
main.main()
