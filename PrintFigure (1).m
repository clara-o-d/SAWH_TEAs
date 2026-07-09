close all
clc
clear

PlotDefaults_Slides()

Y=linspace(0,100);
X=linspace(0,100);

plot(X, Y,'Color',[0.4940, 0.1840, 0.5560])
hold on


set(gcf,'color','w');
xlim([0 100])
ylim([0 100])

alpha=0.7;                                                                  %Scales plot 


fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 5*alpha 4*alpha]; 
print('PlotXXX','-dtiff','-r600')