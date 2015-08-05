
%         RandomGenerator::seed(45423424);

alg='enac';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(1);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='r';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(2);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='rb';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(3);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='g';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(4);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='gb';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(5);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='natr';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(6);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='natrb';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(7);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='natg';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(8);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);
%%
alg='natgb';
A = load(['/tmp/ReLe/lqr/TESTGIRL/objective_',alg,'.log']);
figure(9);
if size(A,2) == 3
    plot3(A(:,1), A(:,2), A(:,3), 'o');
elseif size(A,2) == 2
    plot(A(:,1),A(:,2), 'o');
end
title(alg);
alg = [alg, '1000ep'];
dim = size(A,2);
matlab2tikz(['/home/mpirotta/Dropbox/EWRL2015_irl/tests/objectiveFLQR',num2str(dim),'D/objFLQR',num2str(dim),'D',alg,'.tex']);