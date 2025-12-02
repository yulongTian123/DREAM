clear
data = 'Website Phishing_base_clustering.mat';
load(data);
n = 20;
k = length(unique(gt));
anchor_all = [k-1:1:k+3];
if length(anchor_all) > 5
anchor_all = anchor_all(1:5);
end
alpha_all = [0.9:0.01:1];
for anchor_ind = 1:length(anchor_all)   
for alpha_ind = 1:length(alpha_all)
    alpha = alpha_all(alpha_ind); 
    parfor i= 1:10  %repeat 10 times
        
        [a,b] = size(members);
        zz = RandStream('mt19937ar','Seed',i);
        RandStream.setGlobalStream(zz);
        indx = randperm(b);
        EC_end = members(:,indx(1:n));% Base clustering result
        

        M = Gbe(EC_end);
        [~, mClsLabels] = computeMicroclusters(M);
        [~,Zm] = base_clustering_preserve_dual(mClsLabels,k,anchor_all(anchor_ind));
        simOfCluster = full(simxjac(M'));
        RWofCluster1 = RandomWalkofCluster1(simOfCluster);
        M_new = align_hyper(M,RWofCluster1,alpha);   %hypergraph enhancement
        [F] = solve_DREAM(M_new,anchor_all(anchor_ind),Zm',k);  
        [~,pre_y] = max(F, [], 1);
        result = Clustering5Measure(gt, pre_y); %[ACC nmi Purity AR Entropy]
        ACC(i) = result(1);
        NMI(i) = result(2);
        Purity(i) = result(3);
        ARI(i) = result(4);
        Entropy(i) = result(5);
    end

mean_acc(anchor_ind,alpha_ind)= mean(ACC);
mean_nmi(anchor_ind,alpha_ind) = mean(NMI);
mean_Purity(anchor_ind,alpha_ind) = mean(Purity);
mean_ari(anchor_ind,alpha_ind) = mean(ARI);
mean_Entropy(anchor_ind,alpha_ind) = mean(Entropy);

std_acc(anchor_ind,alpha_ind)= std(ACC);
std_nmi(anchor_ind,alpha_ind) = std(NMI);
std_Purity(anchor_ind,alpha_ind) = std(Purity);
std_ari(anchor_ind,alpha_ind) = std(ARI);
std_Entropy(anchor_ind,alpha_ind) = std(Entropy);
end
end
