## Clinical Information Extraction via Convolutional Neural Network
Code for the paper [Clinical Information Extraction via Convolutional Neural Network](http://arxiv.org/pdf/1603.09381v1.pdf)

1. python processData -input type
2. ./runSpan.sh gpu(cpu) train 20
3. ./runSpan.sh gpu(cpu) test 20
4. th main.lua -device 0(1) 
5. th main.lua -device 0(1) -test 1 -resume (epochs number)
