function lsd = lsderiv(n)
	lsd = logsig(n)*(1 - logsig(n));
    %lsd = gpuArray(answ);
    %what is the period doing here?
    %need to check that this is correct