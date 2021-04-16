function Group = emp_relabel_plot(FE, par, Gr_EST, GIRF, GSE)
% plot graphs and relabel group assignment
% only for the empirical applications in the GLP paper 
H     = par.horizon;
nylag = par.nylag;

LineColors = [  .0  .2  .4];
% [ .2  .4  .6]; %light blue               
          
BandColors =[.7  .7  .7];
         
if FE == 1 && nylag>0
    % FE with lagged dependent variables
    for G0 = 2:5
        if G0 ==2
            Kgrid = [2 1];
        elseif G0 == 3
            Kgrid = [1 2 3];
        elseif G0 == 4
            Kgrid = [4 3 2 1];
        elseif G0 == 5
            Kgrid = [2 5 3 4 1];
        end
        btmp = GIRF{1,G0};
        setmp = GSE{1,G0};
        figure;
        for i = 1:G0
            subplot(ceil(G0/2),2,i);
            k = Kgrid(i);
            hold on;
            ub = btmp(k,:)+1.96*setmp(k,:);
            lb = btmp(k,:)-1.96*setmp(k,:);
            % bands
            fill([1:H, fliplr(1:H)],...
                [ub fliplr(lb)],...
                BandColors,'EdgeColor','none');
            % IR
            plot(1:H, btmp(k,:),'LineWidth',1.2,'color',LineColors);
            
            yline(0,'k','LineWidth',.7);
            xlabel(strcat({'Group'},{' '},num2str(i)));  
            xlim([1 H]); axis tight
            set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
                'FontSize',8,'Layer','top')
            hold off
        end
    end
    
    % relabel them
    Group = Gr_EST(:,2:5);
    tmp = nan(par.N,1);
    tmp(Group(:,1)==2) = 1;
    tmp(Group(:,1)==1) = 2;
    Group(:,1) = tmp;
    
    tmp = nan(par.N,1);
    tmp(Group(:,2)==1) = 1;
    tmp(Group(:,2)==2) = 2;
    tmp(Group(:,2)==3) = 3;
    Group(:,2) = tmp;
    
    tmp = nan(par.N,1);
    tmp(Group(:,3)==4) = 1;
    tmp(Group(:,3)==3) = 2;
    tmp(Group(:,3)==2) = 3;
    tmp(Group(:,3)==1) = 4;
    Group(:,3) = tmp;
    
    tmp = nan(par.N,1);
    tmp(Group(:,4)==2) = 1;
    tmp(Group(:,4)==5) = 2;
    tmp(Group(:,4)==3) = 3;
    tmp(Group(:,4)==4) = 4;
    tmp(Group(:,4)==1) = 5;
    Group(:,4) = tmp;
    
elseif FE ==1 && nylag == 0
    for G0 = 2:4
        if G0 ==2
            Kgrid = [1 2];
        elseif G0 == 3
            Kgrid = [1 2 3];
        elseif G0 == 4
            Kgrid = [1 2 3 4];
        end
        btmp = GIRF{1,G0};
        setmp = GSE{1,G0};
        figure;
        for i = 1:G0
            subplot(ceil(G0/2),2,i);
            k = Kgrid(i);
            hold on;
            ub = btmp(k,:)+1.96*setmp(k,:);
            lb = btmp(k,:)-1.96*setmp(k,:);
            % bands
            fill([1:H, fliplr(1:H)],...
                [ub fliplr(lb)],...
                BandColors,'EdgeColor','none');
            % IR
            plot(1:H, btmp(k,:),'LineWidth',1.2,'color',LineColors);
            
            yline(0,'k','LineWidth',.7);
            xlabel(strcat({'Group'},{' '},num2str(i)));  
            xlim([1 H]); axis tight
            set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
                'FontSize',8,'Layer','top')
            hold off
        end
    end
    
    % relabel them
    Group = Gr_EST(:,2:4);
    
elseif FE ==0 && nylag >0
    
    for G0 = 2:4
        if G0 ==2
            Kgrid = [1 2];
        elseif G0 == 3
            Kgrid = [2 1 3];
        elseif G0 == 4
            Kgrid = [1 3 4 2];
        end
        btmp = GIRF{1,G0};
        setmp = GSE{1,G0};
        figure;
        for i = 1:G0
            subplot(ceil(G0/2),2,i);
            k = Kgrid(i);
            hold on;
            ub = btmp(k,:)+1.96*setmp(k,:);
            lb = btmp(k,:)-1.96*setmp(k,:);
            % bands
            fill([1:H, fliplr(1:H)],...
                [ub fliplr(lb)],...
                BandColors,'EdgeColor','none');
            % IR
            plot(1:H, btmp(k,:),'LineWidth',1.2,'color',LineColors);
            
            yline(0,'k','LineWidth',.7);
            xlabel(strcat({'Group'},{' '},num2str(i)));  
            xlim([1 H]); axis tight
            set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
                'FontSize',8,'Layer','top')
            hold off
        end
    end
    
    % relabel them
    Group = Gr_EST(:,2:4);
    tmp = nan(par.N,1);
    tmp(Group(:,1)==1) = 1;
    tmp(Group(:,1)==2) = 2;
    Group(:,1) = tmp;
    
    tmp = nan(par.N,1);
    tmp(Group(:,2)==2) = 1;
    tmp(Group(:,2)==1) = 2;
    tmp(Group(:,2)==3) = 3;
    Group(:,2) = tmp;
    
    tmp = nan(par.N,1);
    tmp(Group(:,3)==1) = 1;
    tmp(Group(:,3)==3) = 2;
    tmp(Group(:,3)==4) = 3;
    tmp(Group(:,3)==2) = 4;
    Group(:,3) = tmp;
else
    
    for G0 = 2:4
        if G0 ==2
            Kgrid = [1 2];
        elseif G0 == 3
            Kgrid = [2 1 3];
        elseif G0 == 4
            Kgrid = [2 1 3 4];
        end
        btmp = GIRF{1,G0};
        setmp = GSE{1,G0};
        figure;
        for i = 1:G0
            subplot(ceil(G0/2),2,i);
            k = Kgrid(i);
            hold on;
            ub = btmp(k,:)+1.96*setmp(k,:);
            lb = btmp(k,:)-1.96*setmp(k,:);
            % bands
            fill([1:H, fliplr(1:H)],...
                [ub fliplr(lb)],...
                BandColors,'EdgeColor','none');
            % IR
            plot(1:H, btmp(k,:),'LineWidth',1.2,'color',LineColors);
            
            yline(0,'k','LineWidth',.7);
            xlabel(strcat({'Group'},{' '},num2str(i)));  
            xlim([1 H]); axis tight
            set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
                'FontSize',8,'Layer','top')
            hold off
        end
    end
    
    % relabel them
    Group = Gr_EST(:,2:4);
    tmp = nan(par.N,1);
    tmp(Group(:,1)==1) = 1;
    tmp(Group(:,1)==2) = 2;
    Group(:,1) = tmp;
    
    tmp = nan(par.N,1);
    tmp(Group(:,2)==2) = 1;
    tmp(Group(:,2)==1) = 2;
    tmp(Group(:,2)==3) = 3;
    Group(:,2) = tmp;
    
    tmp = nan(par.N,1);
    tmp(Group(:,3)==2) = 1;
    tmp(Group(:,3)==1) = 2;
    tmp(Group(:,3)==3) = 3;
    tmp(Group(:,3)==4) = 4;
    Group(:,3) = tmp;
    
end
end