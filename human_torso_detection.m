% load('./hw2data/trainAnno.mat')

utils = HW2_Utils();
% utils.demo1

[trD, trLb, valD, valLb, trRegs, valRegs] = utils.getPosAndRandomNeg();

% clf = KernelSVM([],[]);
% C = 10;
% clf = clf.train(trD,trLb,C);
% utils.genRsltFile(clf.W,clf.b,'val','./gen_results')
% [ap, prec, rec] = utils.cmpAP('./gen_results', 'val');

[obj_vals,APs] = hard_negative_mining(utils,trD,trLb,valD,valLb,10)

function [obj_vals,avg_precs] = hard_negative_mining(utils,x_train,y_train,x_val,y_val,max_epoch)
    clf = KernelSVM([],[]);
    C = 10;
    obj_vals = zeros(max_epoch,1);
    avg_precs = zeros(max_epoch,1);
    load('./hw2data/trainAnno.mat');
    imFiles = ml_getFilesInDir(sprintf('%s/%sIms/', HW2_Utils.dataDir, 'val), 'jpg');
    for i = 1:max_epoch
        [clf,obj_val,alpha] = clf.train(x_train,y_train,C);
        y_pred = clf.predict(x_val);
        obj_vals(i) = -obj_val;
        fprintf('%d) Obj = %.2f\tVal acc: %.2f\t  Train acc:  %.2f\n',i,-obj_val,clf.get_accuracy(y_pred,y_val),clf.get_accuracy(clf.predict(x_train),y_train))
        fprintf('            \tVal prec: %.2f\t Train prec: %.2f\n',clf.precision(y_pred,y_val),clf.precision(clf.predict(x_train),y_train))
        utils.genRsltFile(clf.W, clf.b, 'val', 'hard_mining');
        [ap,~,~] = utils.cmpAP('hard_mining','val');
        avg_precs(i) = ap;
        non_support_vectors = ~(clf.get_support_vectors(alpha));
        negative_samples = y_train == -1;
        condition = ~(non_support_vectors & negative_samples);
        x_train = x_train(:,condition);
        y_train = y_train(condition,1);
        
        rects = [];
        naya_data = [];
        naye_labels = [];
        for ind = 1:93
            if size(naya_data,2) >=1000
                break
            end
%             if mod(ind,10) == 0
%                 disp(ind)
%             end
            im = imread(sprintf('%s/%sIms/%04d.jpg', utils.dataDir, 'train', ind));
            ubs = ubAnno{ind};
            [h,w,~] = size(im);
            rects = utils.detect(im,clf.W,clf.b,0);
            rects = rects(:,rects(5,:)<0);
%             rects = rects(rects <= h & rects<= h & rects <= w & rects <= w);
            for j=1:size(ubs,2)
                    before = size(rects);
                    overlap = utils.rectOverlap(rects, ubs(:,j));                    
                    rects = rects(:, overlap < 0.3);
                    after = size(rects);
                    if before ~= after
                        disp('overlap hua')
                    end
                    if isempty(rects)
                        break;
                    end
            end
            for r = rects
                if r(2) > h || r(4)> h || r(1) > w || r(3) > w
                    continue
                end
                temp = round(r);
                imReg = im(temp(2):temp(4),temp(1):temp(3),:);  
                imReg = imresize(imReg, HW2_Utils.normImSz);
                imReg_feature = double(HW2_Utils.cmpFeat(rgb2gray(imReg)));
                
                naya_data = [naya_data,imReg_feature];
                naye_labels = [naye_labels;-1];
%                 negD{i} = cat(2, D_i{:});                
%                 negRegs{i} = cat(4, R_i{:});
            end
        end
        naya_data = utils.l2Norm(naya_data);
        x_train = [x_train,naya_data];
        y_train = [y_train;naye_labels];
        

    end
end