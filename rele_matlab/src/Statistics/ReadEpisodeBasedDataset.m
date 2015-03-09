function data = ReadEpisodeBasedDataset(A)

	ds = A(1,1);
	da = A(1,2);
	dr = A(1,3);

	ep = 0;
	for i = 2:size(A,1)
		if (ep == 0 || A(i-1,end) == 1)
			ep = ep + 1;
			data(ep).s  = [];
			data(ep).a  = [];
			data(ep).r  = [];
			data(ep).absorb = [];
		end 
		data(ep).s = [data(ep).s,  A(i, 1:ds)'];
		data(ep).a = [data(ep).a, A(i, ds+1:ds+da)'];
		data(ep).r = [data(ep).r, A(i, ds+da+1:ds+da+dr)'];
		data(ep).absorb = [data(ep).absorb, A(i, end-1)];
	end

return
