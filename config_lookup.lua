--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University
--]]

require("nn")

-- The namespace
config = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

seq_length = 140

-- Training data
config.train_data = {}
config.train_data.file = paths.cwd() .. "/data/type_train.t7b"
config.train_data.alphabet = alphabet
config.train_data.length = seq_length
config.train_data.batch_size = 128

-- Validation data
config.val_data = {}
config.val_data.file =  paths.cwd() .. "/data/type_dev.t7b"
config.val_data.alphabet = alphabet
config.val_data.length = seq_length
config.val_data.batch_size = 128

-- Test data
config.test_data = {}
config.test_data.file =  paths.cwd() .. "/data/type_test.t7b"
config.test_data.alphabet = alphabet
config.test_data.length = seq_length
config.test_data.batch_size = 128

-- The model
config.model = {}
-- #alphabet x 1014
config.model[1] = {module = "nn.LookupTable", char_vocab_size= #alphabet+1, inputFrameSize = 256}
config.model[2] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 6}
config.model[3] = {module = "nn.Threshold"}
config.model[4] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- 336 x 256
config.model[5] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 6}
config.model[6] = {module = "nn.Threshold"}
config.model[7] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- 110 x 256
config.model[8] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[9] = {module = "nn.Threshold"}
-- 108 x 256
config.model[10] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[11] = {module = "nn.Threshold"}
-- 106 x 256
config.model[12] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[13] = {module = "nn.Threshold"}
-- 104 x 256
config.model[14] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[15] = {module = "nn.Threshold"}
config.model[16] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- 34 x 256
config.model[17] = {module = "nn.Reshape", size = 256}
-- 8704
config.model[18] = {module = "nn.Linear", inputSize = 256, outputSize = 128}
config.model[19] = {module = "nn.Threshold"}
config.model[20] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[21] = {module = "nn.Linear", inputSize = 128, outputSize = 128}
config.model[22] = {module = "nn.Threshold"}
config.model[23] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[24] = {module = "nn.Linear", inputSize = 128, outputSize = 4}
config.model[26] = {module = "nn.LogSoftMax"}

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
config.main.eras = 1
config.main.epoches = 5
config.main.randomize = 5e-2
config.main.dropout = true
config.main.save = paths.cwd() .. "/models"
config.main.collectgarbage = 100
config.main.logtime = 5
config.main.validate = true
