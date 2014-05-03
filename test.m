%{
Implementation of Least squares estimation for 10%, 30%, 50% of the random training data on Fisheriris
dataset
The script does the following:
1. Load the data set and split into training and testing parts.
2. Calculate lambda such that the matrix X'T*X' is positive definite.
3. Complete training by calculating W~.
4. Test the data by multiplying weight with the remaining test data.
5. Compute the misclassification error.
%}

%Load Fisheriris dataset
load fisheriris;
p = [10, 30, 50];
totalSpecies = size(species, 1);
miscError = zeros(3, 10);
clStr = unique(species);
nClasses = size(clStr, 1);
graphPlotted = false;
figCnt = 1;

for probLV = (1 : nClasses)
    graphPlotted = false;
    fprintf('Misclassification error for %g percent data \n', p(probLV));
    for loopVar = (1 : 10)
    % Divide into training and testing sets
    [trainInd, testInd, valInd] = dividerand(150, p(probLV)/100, 1 - p(probLV)/100, 0.0);

    % Init
     trainIndSize = size(trainInd, 2);
     testIndSize = size(testInd, 2);         
     trainMeas = zeros(trainIndSize, 4);
     trainSpec = cell(trainIndSize, 1);
     testMeas = zeros(testIndSize, 4);
     testSpec = cell(testIndSize, 1);


    % Training meas and species assignment
      for i = (1 : trainIndSize)              
          trainMeas(i,:) = meas(trainInd(i), :);         
          trainSpec(i) = species(trainInd(i));
      end


    % Testing meas and species assignment
      for i = (1 : testIndSize)    
          testMeas(i,:) = meas(testInd(i), :);         
          testSpec(i) = species(testInd(i));
      end
      
      %Plot data
      if(graphPlotted == false)
            figure(figCnt);
            gscatter(testMeas(:,1), testMeas(:,2), testSpec,'rgb','od*');
            xlabel('Sepal length');
            ylabel('Sepal width');            
            figCnt = figCnt + 1;
      end
        
      %Add one new column to training data      
      newCol = ones(trainIndSize, 1);
      t_trainMeas = [trainMeas newCol];
      x = t_trainMeas;
      %Calculate X'T*X'
      tx = x'*x;
      t_x = tx;


      %TODO : Set lambda value
      EPS = 10^-6;      %SET THE VALUE TO PLACE INSTEAD OF ZERO OR NEGATIVE %EIGENVALUES  
      ZERO = 10^-10;    %SET THE VALUE TO LOOK FOR    
      [~, err] = cholcov(t_x, 0); %ERR<>0 MEANS SOME EIGENVALUES <=0 
      if (err ~= 0) 
        [v, d] = eig(t_x); %CALCULATE EIGENVECTOR AND EIGENVALUES 
        d=diag(d);          %GET A VECTOR OF EIGENVALUES 
        d(d<=ZERO)=EPS; %FIND ALL EIGENVALUES<=ZERO AND CHANGE THEM FOR EPS 
        d=diag(d);      %CONVERT VECTOR d INTO A MATRIX 
        t_x = v*d*v'; %RECOMPOSE t_x MATRIX USING EIGENDECOMPOSITION 
                        %WHY? t_x IS SIMETRIC AND V IS ORTHONORMAL 
      end 

      %Formulate T
      T = [];      
      for i = (1 : trainIndSize)
          row = [0 0 0];
          [f, fIdx] = ismember(trainSpec(i), clStr);
          if(f == 1)
            row(fIdx) = 1;
          end
          T = [T; row];
      end

      %Calculate Weight
      W = (t_x) \ (x'* T);

      newCol = ones(testIndSize, 1);
      t_testMeas = [testMeas newCol]';
      x2 = t_testMeas;
      R = W' * x2;
      R = R';

      %Test classification
      testClassification = cell(testIndSize, 1);
      for i = (1 : testIndSize)
          maxVal = max(R(i, :));
          idx = 0;
          for j = (1 : size(clStr, 1))
            if(maxVal == R(i, j))
                %fprintf('setting %g\n', j);
                idx = j;
            end
          end
          if(idx ~= 0)
              testClassification(i) = clStr(idx);
          end
      end

      %Histogram plot
      if(graphPlotted == false)                
          figure(figCnt);
          gscatter(testMeas(:,1), testMeas(:,2), testClassification,'rgb','od*');
          xlabel('Sepal length');
          ylabel('Sepal width');            
          graphPlotted = true;
          figCnt = figCnt + 1;
      end        
        
      %Misclassification error calculations
      mcMat = ~cellfun(@strcmp, testSpec, testClassification);
      mError = sum(mcMat)/size(species, 1);  
      miscError(probLV, loopVar) = mError;
      fprintf('%g\n', mError);  
    end
end

for i=(1 : nClasses)
    fprintf('Average misclassification error for %g percent data %g\n', p(i), sum(miscError(i,:))/10);
end