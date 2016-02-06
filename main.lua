require("data")
require("model")
require("train")
require("test")

dofile("config.lua")

-- Prepare random number generator
math.randomseed(os.time())
torch.manualSeed(os.time())


main = {}

function main.main()

	main.clock = {}
   	main.clock.log = 0

	main.argparse()
	main.new()
	main.run()
end

-- Parse arguments
function main.argparse()
   local cmd = torch.CmdLine()
   -- Options
   cmd:option("-debug",0, "debug setting")
   cmd:option("-gpu", 0, "using gpu")
   cmd:text()
   
   -- Parse the option
   local opt = cmd:parse(arg or {})   
   if opt.debug > 0 then
		dbg = require('debugger')
   end

	if opt.gpu > 0 then
   		require("cutorch")
   		require("cunn")
   		cutorch.setDevice(opt.gpu)
   		print("Device gpu set to ".. opt.gpu)
   	else
   		print("Device is cpu")
	end

end

function main.new()
	print("Loading datasets...")
	main.train_data = Data(config.train_data)
	main.val_data = Data(config.val_data)

	print("Loading the model...")
	main.model = Model(config.model, config.main.type)
	if config.main.randomize then
		main.model:randomize(config.main.randomize)
		print("Model randomized.")
	end
	main.model:type(config.main.type)
	print("Current model type: "..main.model:type())
	collectgarbage()

	print("Loading the trainer...")
	main.train = Train(main.train_data, main.model, config.loss(), config.train)

	print("Loading the tester...")
	main.test_val = Test(main.val_data, main.model, config.loss(), config.test)

	collectgarbage()
end

function main.run()

	for i = 1, config.main.eras do
		if config.main.dropout then
			print("Enabling dropouts")
			main.model:enableDropouts()
		else
			print("Disabling dropouts")
			main.model:disableDropouts()
		end

		print("Training for era " .. i)
		main.train:run(config.main.epoches, main.trainlog)

		if config.main.test == true then
			print("Disabling dropouts")
			main.model:disableDropouts()
			print("Testing on test data for era " .. i)
  			main.test_val:run(main.testlog)
  		end

  		print("val_error is"..string.format("%.2e",main.test_val.e))

  	end
end

-- The training logging function
function main.trainlog(train)
   if config.main.collectgarbage and math.fmod(train.epoch-1,config.main.collectgarbage) == 0 then
      print("Collecting garbage at epoch = "..(train.epoch-1))
      collectgarbage()
   end

   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      local msg = ""
      
      if config.main.details then
	 msg = msg.."epo: "..(train.epoch-1)..
	    ", rat: "..string.format("%.2e",train.rate)..
	    ", err: "..string.format("%.2e",train.error)..
	    ", obj: "..string.format("%.2e",train.objective)..
	    ", dat: "..string.format("%.2e",train.time.data)..
	    ", fpp: "..string.format("%.2e",train.time.forward)..
	    ", bpp: "..string.format("%.2e",train.time.backward)..
	    ", upd: "..string.format("%.2e",train.time.update)
      end
      
      if config.main.debug then
	 msg = msg..", bmn: "..string.format("%.2e",train.batch:mean())..
	    ", bsd: "..string.format("%.2e",train.batch:std())..
	    ", bmi: "..string.format("%.2e",train.batch:min())..
	    ", bmx: "..string.format("%.2e",train.batch:max())..
	    ", pmn: "..string.format("%.2e",train.params:mean())..
	    ", psd: "..string.format("%.2e",train.params:std())..
	    ", pmi: "..string.format("%.2e",train.params:min())..
	    ", pmx: "..string.format("%.2e",train.params:max())..
	    ", gmn: "..string.format("%.2e",train.grads:mean())..
	    ", gsd: "..string.format("%.2e",train.grads:std())..
	    ", gmi: "..string.format("%.2e",train.grads:min())..
	    ", gmx: "..string.format("%.2e",train.grads:max())..
	    ", omn: "..string.format("%.2e",train.old_grads:mean())..
	    ", osd: "..string.format("%.2e",train.old_grads:std())..
	    ", omi: "..string.format("%.2e",train.old_grads:min())..
	    ", omx: "..string.format("%.2e",train.old_grads:max())
	 main.draw()
      end
      
      if config.main.details or config.main.debug then
	 print(msg)
      end

      main.clock.log = os.time()
   end
end

function main.testlog(test)
   if config.main.collectgarbage and math.fmod(test.n,config.train_data.batch_size*config.main.collectgarbage) == 0 then
      print("Collecting garbage at n = "..test.n)
      collectgarbage()
   end
   if not config.main.details then return end
   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      print("n: "..test.n..
	       ", e: "..string.format("%.2e",test.e)..
	       ", l: "..string.format("%.2e",test.l)..
	       ", err: "..string.format("%.2e",test.err)..
	       ", obj: "..string.format("%.2e",test.objective)..
	       ", dat: "..string.format("%.2e",test.time.data)..
	       ", fpp: "..string.format("%.2e",test.time.forward)..
	       ", acc: "..string.format("%.2e",test.time.accumulate))
      main.clock.log = os.time()
   end
end

--Execute the main program
main.main()