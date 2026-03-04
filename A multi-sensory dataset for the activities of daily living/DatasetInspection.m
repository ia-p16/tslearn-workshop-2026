volunteers = dir("./volunteer*");
series1 = {"DrinkGlass","PourWater","StandUp","Walk","OpenDoor","CleanTable","Walk","SitDown","PourWater","DrinkGlass","StandUp","Walk","OpenDoor","BrushTeeth","Walk","CleanTable","SitDown","DrinkGlass","PourWater","StandUp","Walk","CloseDoor","BrushTeeth","Walk","CleanTable","CloseDoor","Walk","Sit","StandUp","PourWater","CleanTable","DrinkGlass","Walk","SitDown","CleanTable","StandUp","Walk","BrushTeeth","Walk","CloseDoor","Walk","BrushTeeth","OpenDoor","BrushTeeth","CloseDoor"}';
series2 = {"StandUp","CleanTable","SitDown","PourWater","StandUp","DrinkGlass","Walk","DrinkGlass","OpenDoor","CloseDoor","PourWater","Walk","PourWater","OpenDoor","PourWater","Walk","OpenDoor","DrinkGlass","BrushTeeth","PourWater","CloseDoor","DrinkGlass","Walk","DrinkGlass","CloseDoor","CleanTable","PourWater","SitDown","DrinkGlass","CleanTable","PourWater","StandUp","Walk","OpenDoor","PourWater","BrushTeeth","DrinkGlass","Walk","CleanTable","OpenDoor","Walk","DrinkGlass","SitDown","StandUp","Walk","CloseDoor"}';
activities = {"Walk","SitDown","StandUp","OpenDoor","CloseDoor", "PourWater", "DrinkGlass", "BrushTeeth", "CleanTable"};
sensors = {"rua","rt","rla","lua","lla","back"};
total_experiment = [series1;series1;series2;series2];
exp_activities_distribution = zeros(1,length(activities));

for i=1:length(activities)
   exp_activities_distribution(1,i) =  length(find(table2array(cell2table(total_experiment)) == activities{i}));
end


activities_distribution = zeros(length(volunteers),length(activities));
for i=1:length(volunteers)
    T = readtable(strcat("./",volunteers(i).name,"/annotations.csv"));
    start_instants = find(table2array(T(:,4)) == "Start");
    start_time = table2array(T(start_instants,2));
    end_instants = find(table2array(T(:,4)) == "End");
    end_time = table2array(T(end_instants,2));
    labels = {"ADL Label"};
    labels = [labels;table2array(T(start_instants,3))];
    
    
    for j=1:length(sensors)
        data_table = readtable(strcat("./",volunteers(i).name,"/IMUs/",sensors{j},".csv"));
        stamps = table2array(data_table(:,2));
        [minValue, closestIndex] = min(abs(stamps - start_time.'));
        temp = [strcat(sensors{j},"_start"); num2cell(stamps(closestIndex))];
        labels = [labels, temp];
        [minValue, closestIndex] = min(abs(stamps - end_time.'));
        temp = [strcat(sensors{j},"_end"); num2cell(stamps(closestIndex))];
        labels = [labels, temp];
    end
    
    experiment{i,1} = volunteers(i).name;
    experiment{i,2} = labels;
    
    for j=1:length(activities)
        activities_distribution(i,j) =  length(find(labels(:,1) == activities{j}));
    end
end

activities_distribution - exp_activities_distribution