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
require("gnuplot")

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

      print("Recording on ear " .. i)
      main.record[#main.record + 1] = {val_error = main.test_val.e, val_loss = main.test_val.l}
      print("Visualizing loss")
      main.show()
      main.save()
      collectgarbage()
   end
end

function main.show(figure_error, figure_loss)
   main.figure_error = main.figure_error or gnuplot.figure()
   main.figure_loss = main.figure_loss or gnuplot.figure()

   local figure_error = figure_error or main.figure_error
   local figure_loss = figure_loss or main.figure_loss

   local epoch = torch.linspace(1, #main.record, #main.record):mul(config.main.epoches)
   local val_error = torch.zeros(#main.record)
   local val_loss = torch.zeros(#main.record)
   for i = 1, #main.record do
      val_error[i] = main.record[i].val_error
      val_loss[i] = main.record[i].val_loss
   end

   print("val_error is")
   print(val_error)

   --Do the plot
   gnuplot.figure(figure_error)
   gnuplot.plot({"Validate", epoch, val_error})
   gnuplot.title("Validating error")
   gnuplot.plotflush()
   gnuplot.figure(figure_loss)
   gnuplot.plot({"Validate", epoch, val_loss})
   gnuplot.title("Validating loss")
   gnuplot.plotflush()

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

   local preds_out = torch.DiskFile(paths.cwd() .. "/" .. att_name .."_decisions.txt", 'w')
   for i = 1, preds:size(1) do
    preds_out:writeString(preds[i] .. "\n")
   end
   print("predicted events is: " .. preds:size(1)) -- 96326
   preds_out:close()

   os.execute("python makeEvaluation.py")

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

   main.eps_error = main.eps_error or gnuplot.epsfigure(paths.concat(config.main.save,"figure_error.eps"))
   main.eps_loss = main.eps_loss or gnuplot.epsfigure(paths.concat(config.main.save,"figure_loss.eps"))
   main.show(main.eps_error, main.eps_loss)

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
