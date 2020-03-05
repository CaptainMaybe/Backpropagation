function plotOutcomes()
%     testBP_100 = [backPropTesting(0, 100) backPropTesting(4, 100) backPropTesting(8, 100)]';
%     testBP_1000 = [backPropTesting(0, 1000) backPropTesting(4, 1000) backPropTesting(8, 1000)]';
%     testBP_10000 = [backPropTesting(0, 10000) backPropTesting(4, 10000) backPropTesting(8, 10000)]';
% %    subplot(1, 2, 1)
%     testBP = [testBP_100 testBP_1000 testBP_10000];
%     figure(1)
%     plot(testBP,'-x','LineWidth',2, 'MarkerSize',10);
%     xlabel('Number of noisy pixels')
%     ylabel('Mean Squared Error')
%     title('Correctness vs Number of Noisy Pixels using BackPropagation')
%      legend({'100 epochs','1,000 epochs', '10,000 epochs'},'Location', 'northwest')
%      set(gca,'XTick', [1 2 3])
%      set(gca,'XTickLabel', [ 0 4 8])
% %     xlim([0 8])
% %     ylim([0 100.09])
%      xticks([1 2 3])
%      grid on
    

%--------------------------------------------------------------------------
    learningRates = [0.1 0.5 1 2 5 10];
    outputs = zeros(1, 6);
    for i = 1:length(learningRates)
        outputs(i) = numberRecognizer(learningRates(i));
    end
    
    figure(1)
    plot(outputs,'-x','LineWidth',2, 'MarkerSize',10);
    xlabel('Learning Rate')
    ylabel('Mean Squared Error')
    title('Correctness vs Choice of Learning Rate')
%      legend({'100 epochs','1,000 epochs', '10,000 epochs'},'Location', 'northwest')
     set(gca,'XTick', [1 2 3 4 5 6 ])
     set(gca,'XTickLabel', [ 0.1 0.5 1 2 5 10])
     xlim([0 6.1])
%     ylim([0 100.09])
      xticks([1 2 3 4 5 6 ])
     grid on


%-------------------------------------------------------------------------
%     figure(2)
%     %subplot(1, 2, 2)
%     testInv = pseudoInverse(2)';
%     testInv = [testInv pseudoInverse(4)'];
%     testInv = [testInv pseudoInverse(6)'];
%     plot(testInv,'LineWidth',2);
%     xlabel('Number of different input vectors')
%     ylabel('Percent Error')
%     %title('Correctness over increasing numbers of input vectors using Pseudo Inverse Rule')
%     legend({'2 pixel errors','4 pixel errors', '6 pixel errors'},'Location', 'northwest')
%     xlim([1 7])
%     ylim([0 100.09])
%     xticks([2 3 4 5 6 7])
%     grid on