classdef csvWrapper<handle
    %CSVWRAPPER Class to wrap a csv file
    %   Detailed explanation goes here
    
    properties
        index
        batchSize
        lastBatch
        fileId
        nRows
    end
    
    methods
        function obj = csvWrapper(fileName)
            obj.index = 0;
            obj.batchSize = 6000000;
            obj.fileId = fopen(fileName);
            R = textscan(obj.fileId,'%1c%*[^\n]', 'CollectOutput', true);
            obj.nRows = numel(R{1, 1});
            frewind(obj.fileId);
            
        end
        
        function s = size(obj)
            s = obj.nRows;
        end
        
        function l = readLine(obj, i, varargin)
            if i > obj.index
                batchCell = readCSVLines(obj.fileId, obj.batchSize);
                obj.lastBatch = batchCell{1};
                obj.index = obj.index + obj.batchSize;
            end
            
            switch(nargin)
                
                case 2
                    batchIndex = mod(i - 1, obj.batchSize) + 1;
                    l = obj.lastBatch(batchIndex, 1:end);
                    
                case 3
                    s = varargin{1};
                    batchIndex = mod(i - 1, obj.batchSize) + 1;
                    l = obj.lastBatch(batchIndex, s);
                    
                case 4
                    s = varargin{1};
                    e = varargin{2};
                    batchIndex = mod(i - 1, obj.batchSize) + 1;
                    l = obj.lastBatch(batchIndex, s:e);
            end
            
            
            
        end
        
        
    end
    
end

