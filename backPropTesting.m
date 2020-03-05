function answer = backPropTesting(numRandomErrors, num_epochs)
    p0 = [-1 1 1 1 1 -1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 1 -1 1 1 1 1 -1]';
    p1 = [-1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]';
    p2 = [1 -1 -1 -1 -1 -1 1 -1 -1 1 1 1 1 -1 -1 1 -1 1 -1 1 1 -1 -1 1 -1 -1 -1 -1 -1 1]';
    %combine p inputs
    AllP = [p0 p1 p2];

    t0 = [1 0 0]';
    t1 = [0 1 0]';
    t2 = [0 0 1]';
    T = [t0 t1 t2];
    correct = 0;
    total = 0;
    
    for p = 1: 3
        [W1, W2, b1, b2, out] = backprop(num_epochs);
        for i = 1:10 %testing 10 times
            noisyP = [addNoise(AllP(:,p), numRandomErrors)];
            [n1, n2, a, a1] = propogateForward(noisyP, W1, W2, b1, b2);
            e = sum(squaredError(a, T(:,p)));
            total = total + e;
        end
    end
    meanSquaredError = total/30;
%     imagesc(W1);
    %imagesc(W2);
%     colormap(hsv);
%     colorbar;
 answer = meanSquaredError;