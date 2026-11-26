function history=evaluate_iteration_history(data,result,output_dir)
% Post-estimation diagnostics only. This function never controls iteration.
if ~isfield(data,'truth'), error('STAR_KF:IterationTruth','Evaluation truth is required after estimation.'); end
ref=data.i_master; n=data.n_slc; keep=setdiff(1:n,ref);
truth=data.truth.phi_def_mm; truth=truth-truth(:,ref);
truth_atm=data.truth.phi_turb_mm; truth_atm=truth_atm-truth_atm(:,ref);
active=max(abs(truth),[],2)>1e-8;
if ~any(active)||all(active), error('STAR_KF:EvaluationRegions','Both activity regions required by this diagnostic.'); end
B=zeros(data.n_ifg,n);
for k=1:data.n_ifg, B(k,data.ifg_pairs(k,:))=[-1,1]; end
truth_ifg=truth*B'; truth_atm_ifg=truth_atm*B'; scale=data.scale_rad2mm;
N=numel(result.iterations); rows=cell(2*N,1);
for q=1:N
    current=result.iterations{q};
    for stage=1:2
        if stage==1
            label='outer_end'; source='same_round_temporal_stage'; fit=current;
            residual=current.E_final_q*scale;
        elseif q<N
            label='final_if_stop'; source='next_round_temporal_stage'; fit=result.iterations{q+1};
            residual=(data.phi_mat-current.tau_q)*scale-fit.D_joint_q*B';
        else
            label='final_if_stop'; source='actual_finalizer'; fit=result.finalization.deformation;
            fit.Y_q=result.timeseries_atm_corrected;
            fit.W2=fit.weights.W2;
            residual=result.residual_final*scale;
        end
        d=fit.D_joint_q-truth; e1=fit.D1_q-truth; ey=fit.Y_q-truth;
        ea=current.a_q*scale-truth_atm; et=current.tau_q*scale-truth_atm_ifg;
        assert(all(isfinite([d(:);e1(:);ey(:);ea(:);et(:)])));
        row=struct('round',q,'stage',label,'source',source, ...
            'joint_rmse_mm',rmsall(d(:,keep)),'joint_mae_mm',meanall(abs(d(:,keep))), ...
            'joint_bias_mm',meanall(d(:,keep)),'joint_p95_abs_mm',prctile(abs(reshape(d(:,keep),[],1)),95), ...
            'active_rmse_mm',rmsall(d(active,keep)),'active_mae_mm',meanall(abs(d(active,keep))), ...
            'background_rmse_mm',rmsall(d(~active,keep)),'background_mae_mm',meanall(abs(d(~active,keep))), ...
            'final_epoch_rmse_mm',rmsall(d(:,end)),'final_epoch_mae_mm',meanall(abs(d(:,end))), ...
            'kf1_rmse_mm',rmsall(e1(:,keep)),'kf1_mae_mm',meanall(abs(e1(:,keep))), ...
            'network_rmse_mm',rmsall(ey(:,keep)),'network_mae_mm',meanall(abs(ey(:,keep))), ...
            'atmosphere_epoch_rmse_mm',rmsall(ea(:,keep)),'atmosphere_epoch_mae_mm',meanall(abs(ea(:,keep))), ...
            'atmosphere_ifg_rmse_mm',rmsall(et),'atmosphere_ifg_mae_mm',meanall(abs(et)), ...
            'corrected_ifg_rmse_mm',rmsall((data.phi_mat-current.tau_q)*scale-truth_ifg), ...
            'model_residual_rmse_mm',rmsall(residual), ...
            'mean_acceptance_weight',mean(fit.W2), ...
            'Delta_D',current.Delta_D,'Delta_tau',current.Delta_tau,'Delta_R',current.Delta_R, ...
            'Delta_E',current.Delta_E,'C',current.convergence.C_diagnostic, ...
            'criterion_fulfilled',current.convergence.terminate);
        rows{2*q-2+stage}=row;
    end
end
history=vertcat(rows{:});
if nargin>=3 && ~isempty(output_dir)
    if ~isfolder(output_dir), mkdir(output_dir); end
    writetable(struct2table(history),fullfile(output_dir,'iteration_accuracy.csv'));
    fid=fopen(fullfile(output_dir,'iteration_accuracy_matlab.json'),'w','n','UTF-8');
    cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s',jsonencode(history,'PrettyPrint',true));
end
end
function v=rmsall(x), v=sqrt(mean(x(:).^2)); end
function v=meanall(x), v=mean(x(:)); end
