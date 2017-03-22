function figureQSS
% a figure a quarter of the screen size
    scsz = get(0,'ScreenSize'); % scsz = [left botton width height]
    figure('Position',[scsz(3)/4 scsz(4)/4 scsz(3)/2 scsz(4)/2])
end