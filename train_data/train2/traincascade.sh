opencv_traincascade -data /Volumes/Macintosh/Users/zBritva/Projects/LandmarkRecognition/train_data/train2/trained/ -vec train.vec -bg bg.txt -numPos 800 -numNeg 113 -numStages 16 -precalcValBufSize 2048 -precalcIdxBufSize 4096 -numThreads 5 -featureType LBP -w 40 -h 40 -acceptanceRatioBreakValue 10e-5 -maxFalseAlarmRate 0.4 -mode ALL