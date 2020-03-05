function [w1, w2, b1, b2, out] = backprop(epoch_num, P, T,r)

%     p0 = [-1 1 1 1 1 -1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 1 -1 1 1 1 1 -1]';
%     p1 = [-1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]';
%     p2 = [1 -1 -1 -1 -1 -1 1 -1 -1 1 1 1 1 -1 -1 1 -1 1 -1 1 1 -1 -1 1 -1 -1 -1 -1 -1 1]';
% 
%     %combine p inputs
%     P = [p0 p1 p2];

%     t0 = [1 0 0]';
%     t1 = [0 1 0]';
%     t2 = [0 0 1]';
%     T = [t0 t1 t2];
    %each p is a 30x1 matrix, therefore the weight must be a 1x30 matrix

    numNeuronInputsW1 = length(P(:,1)); %num of inputs for each input vector found by taking the length of the first column
    numNeuronOutputsW1 = 20; %can play around with this number for the number of neurons in the hidden layer
    numNueronOutputsW2 = length(T(:,1));

    %randomize the starting weights to values close to zero
    w1 = 2.*rand(numNeuronInputsW1,numNeuronOutputsW1).' - 1;
    w2 = 2.*rand(numNeuronOutputsW1, numNueronOutputsW2).' - 1;
    b1 = 2.*rand(numNeuronOutputsW1,1) - 1;
    b2 = 2.*rand(numNueronOutputsW2,1) - 1;

    % [n1, n2, a, a1] = propogateForward(p0, w1, w2, b1, b2);
    % s2 = computeOutputSensitivities(a, t0, n2);
    % s1 = backpropogateSensitivities(w2, s2, n1);
    % [w1, b1] = updateWeightAndBiases(w1, b1, s1, p0);
    % [w2, b2] = updateWeightAndBiases(w2, b2, s2, a1);

    input_num = length(P(1,:)); %number of input vectors taken by finding the length of the first row of P
    

    for j = 1:epoch_num
      for i = 1:input_num %i is column num (or respective input vector)
        [n1, n2, out, a1] = propogateForward(P(:,i), w1, w2, b1, b2);
        
        s2 = computeOutputSensitivities(out, T(:,i), n2);
        s1 = backpropogateSensitivities(w2, s2, n1);
        [w1, b1] = updateWeightAndBiases(w1, b1, s1, P(:,i),r);
        [w2, b2] = updateWeightAndBiases(w2, b2, s2, a1,r);
      end
    end