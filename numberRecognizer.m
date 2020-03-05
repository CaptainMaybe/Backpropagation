function [meanSquaredError] = numberRecognizer(r)
    %W1, W2, b1, b2, out
    % Change the filenames if you have saved the files under different names
    % On some platforms, the files might be saved as 
    % train-images.idx3-ubyte / train-labels.idx1-ubyte

    images = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
   
    P = images;
    T = convertT(labels)';
%     for i = 1: 60000
%         t = convertT(labels(i));
%         T(:,i) = t';%set each column of the T matrix to the converted label vector
%     end
    [W1, W2, b1, b2, out] = backprop(10,P, T,r);
    
    test_im = loadMNISTImages('t10k-images.idx3-ubyte');
    test_la = loadMNISTLabels('t10k-labels.idx1-ubyte');
   
    testMatrix = convertT(test_la)';
    total = 0;
    totalSquaredError = 0;
    
    for p = 1: length(test_im(1,:))
        [n1, n2, a, a1] = propogateForward(test_im(:,p), W1, W2, b1, b2);
        e = sum(squaredError(a, testMatrix(:,p)));
        totalSquaredError = totalSquaredError + e;
       
    end
    meanSquaredError = totalSquaredError/10000.0;