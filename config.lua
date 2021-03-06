--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University
--]]

require("nn")

-- The namespace
config = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

seq_length = 100
att_name = "type"
att_num_label = 4

-- Training data
config.train_data = {}
config.train_data.file = paths.cwd() .. "/data/".. att_name .. "_train.t7b"
config.train_data.alphabet = alphabet
config.train_data.length = seq_length
config.train_data.batch_size = 128

-- Validation data
config.val_data = {}
config.val_data.file =  paths.cwd() .. "/data/".. att_name .. "_dev.t7b"
config.val_data.alphabet = alphabet
config.val_data.length = seq_length
config.val_data.batch_size = 128

-- Test data
config.test_data = {}
config.test_data.file =  paths.cwd() .. "/data/".. att_name .. "_test.t7b"
config.test_data.alphabet = alphabet
config.test_data.length = seq_length
config.test_data.batch_size = 128

-- The model
config.model = {}
-- #alphabet x 1014
config.model[1] = {module = "nn.TemporalConvolution", inputFrameSize = #alphabet, outputFrameSize = 256, kW = 5}
config.model[2] = {module = "nn.Threshold"}
config.model[3] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}
-- 336 x 256
config.model[4] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 5}
config.model[5] = {module = "nn.Threshold"}
config.model[6] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}
-- 110 x 256
config.model[7] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[8] = {module = "nn.Threshold"}
-- 108 x 256
config.model[9] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[10] = {module = "nn.Threshold"}
-- 106 x 256
config.model[11] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[12] = {module = "nn.Threshold"}
-- 104 x 256
config.model[13] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[14] = {module = "nn.Threshold"}
config.model[15] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- 34 x 256
config.model[16] = {module = "nn.Reshape", size = 1024}
-- 8704
config.model[17] = {module = "nn.Linear", inputSize = 1024, outputSize = 256}
config.model[18] = {module = "nn.Threshold"}
config.model[19] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[20] = {module = "nn.Linear", inputSize = 256, outputSize = 256}
config.model[21] = {module = "nn.Threshold"}
config.model[22] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[23] = {module = "nn.Linear", inputSize = 256, outputSize = att_num_label}
config.model[24] = {module = "nn.LogSoftMax"}

-- The loss
config.loss = nn.ClassNLLCriterion

-- The trainer
config.train = {}
local baseRate = 1e-2 * math.sqrt(config.train_data.batch_size) / math.sqrt(128)
config.train.rates = {[1] = baseRate/1,[15001] = baseRate/2,[30001] = baseRate/4,[45001] = baseRate/8,[60001] = baseRate/16,[75001] = baseRate/32,[90001]= baseRate/64,[105001] = baseRate/128,[120001] = baseRate/256,[135001] = baseRate/512,[150001] = baseRate/1024}
config.train.momentum = 0.9
config.train.decay = 1e-5

-- The tester
config.test = {}
config.test.confusion = true


-- Main program
config.main = {}
config.main.eras = 10
config.main.epoches = 20
config.main.randomize = 5e-2
config.main.dropout = true
config.main.save = paths.cwd() .. "/models_" ..att_name
config.main.collectgarbage = 100
config.main.logtime = 5
config.main.validate = true
