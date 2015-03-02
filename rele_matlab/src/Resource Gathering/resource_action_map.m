% Plots a map of the distribution of each action on the environment.
% NaN is used as delimitation in the matrices.

policy = gibbs_policy(@resource_basis_pol,theta,[1,2,3,4]);

map1 = cell(5);
map2 = cell(5);

for i = 1 : 5
    for j = 1 : 5
        distrib = zeros(4,1);
        state = [i; j; 0; 0];
        for action = 1 : 4
            distrib(action) = policy.evaluate(state,action);
        end
        map1{i,j} = reshape(distrib,2,2)';
        state = [i; j; 0; 1];
        for action = 1 : 4
            distrib(action) = policy.evaluate(state,action);
        end
        map2{i,j} = reshape(distrib,2,2)';
    end
end

disp1 = cell2mat(map1);
delimiter = NaN(1,size(disp1,1));
disp1 = [disp1(1:2,:); delimiter; disp1(3:4,:); delimiter; disp1(5:6,:); delimiter; disp1(7:8,:); delimiter; disp1(9:10,:)];
delimiter = NaN(size(disp1,1),1);
disp1 = [disp1(:,1:2) delimiter disp1(:,3:4) delimiter disp1(:,5:6) delimiter disp1(:,7:8) delimiter disp1(:,9:10)];
disp2 = cell2mat(map2);
delimiter = NaN(1,size(disp2,1));
disp2 = [disp2(1:2,:); delimiter; disp2(3:4,:); delimiter; disp2(5:6,:); delimiter; disp2(7:8,:); delimiter; disp2(9:10,:)];
delimiter = NaN(size(disp2,1),1);
disp2 = [disp2(:,1:2) delimiter disp2(:,3:4) delimiter disp2(:,5:6) delimiter disp2(:,7:8) delimiter disp2(:,9:10)];
