clear
data = 'Website Phishing_base_clustering.mat';
load(data);
n = 20;
k = length(unique(gt));
anchor_all = [k-1,k:1:k+3];
if length(anchor_all) > 5
anchor_all = anchor_all(1:5);
end
alpha_all = [0.9:0.01:1];
for ia = 1:length(anchor_all)   
    tic
for alpha_i = 1:length(alpha_all)
    alpha = alpha_all(alpha_i);
    
parfor i= 1:10
    
    [a,b] = size(members);
    zz = RandStream('mt19937ar','Seed',i);
    RandStream.setGlobalStream(zz);
    indx = randperm(b);
    EC_end = members(:,indx(1:n));% Base clustering result
    M = Gbe(EC_end);
    para_theta = 0.4;
    [~, mClsLabels] = computeMicroclusters(M);
    [YY_label,Zm] = base_clustering_preserve_dual(mClsLabels,k,anchor_all(ia));
    simOfCluster = full(simxjac(M'));
    RWofCluster1 = RandomWalkofCluster(simOfCluster);
    M_new = align_hyper(M,RWofCluster1,alpha);
    consensus_ECI = computeECI_hyper(M_new, para_theta,n);
    [U,~] = solve_ECADH_journal(M_new,anchor_all(ia),consensus_ECI',YY_label,Zm);
    stream = RandStream.getGlobalStream;
    reset(stream);
    U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,size(U,2));
    indx = kmeans(U_normalized,k,'MaxIter',100, 'Replicates',3);
    result = Clustering5Measure(gt, indx); %[ACC nmi Purity AR Entropy]
    ACC(i) = result(1);
    NMI(i) = result(2);
    Purity(i) = result(3);
    ARI(i) = result(4);
    Entropy(i) = result(5);
end

mean_acc(ia,alpha_i)= mean(ACC);
mean_nmi(ia,alpha_i) = mean(NMI);
mean_Purity(ia,alpha_i) = mean(Purity);
mean_ari(ia,alpha_i) = mean(ARI);
mean_Entropy(ia,alpha_i) = mean(Entropy);


std_acc(ia,alpha_i)= std(ACC);
std_nmi(ia,alpha_i) = std(NMI);
std_Purity(ia,alpha_i) = std(Purity);
std_ari(ia,alpha_i) = std(ARI);
std_Entropy(ia,alpha_i) = std(Entropy);
end
toc
end
[max_acc] = max(max(mean_acc));
[max_nmi] = max(max(mean_nmi));
[max_ari] = max(max(mean_ari));
[max_Purity] = max(max(mean_Purity));
[max_Entropy] = max(max(mean_Entropy));


[idx,idy] = find(mean_acc==max_acc);
std_accia = std_acc(idx(1,1),idy(1,1));

[idx,idy] = find(mean_nmi==max_nmi);
std_nmiia = std_nmi(idx(1,1),idy(1,1));

[idx,idy] = find(mean_ari==max_ari);
std_ariia = std_ari(idx(1,1),idy(1,1));

 [idx,idy] = find(mean_Purity==max_Purity);
 std_Purityia = std_Purity(idx(1,1),idy(1,1));

 [idx,idy] = find(mean_Entropy==max_Entropy);
 std_Entropyia = std_Entropy(idx(1,1),idy(1,1));

%RE=zeros(4,2);
RE(1,1) = max_acc;
RE(2,1) = max_ari;
RE(3,1) = max_nmi;
RE(4,1) = max_Purity;
RE(5,1) = max_Entropy;

RE(1,2) = std_accia;
RE(2,2) = std_ariia;
RE(3,2) = std_nmiia;
RE(4,2) = std_Purityia;
RE(5,2) = std_Entropyia;
