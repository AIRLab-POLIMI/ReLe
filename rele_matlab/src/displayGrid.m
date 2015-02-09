function displayGrid(policy)

for i=1:size(policy, 2)
    switch(policy(i))
        case 0
            fprintf('^')
        case 1
            fprintf('v')
        case 2
            fprintf('<')
        case 3
            fprintf('>')
    end
   
   if(mod(i, 8) == 0)
       fprintf('\n');
   end
end

end