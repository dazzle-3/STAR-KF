function tests=test_kf2_acceptance
tests=functiontests(localfunctions);
end
function testThreeComponentsAndAmplitudeCaps(testCase)
cfg=star_kf_v4_config(); n=10; ref=4;
T=eye(n); T(:,ref)=T(:,ref)-1; RE=T*T'; C2=.01*RE;
e=[0,-2,-5,0,-8,-12,-7,-2,3,7]; d=.5*e;
[w,diagnostic]=build_kf2_acceptance_weights(d,C2,e,RE,ref,cfg.kf2_acceptance);
verifyEqual(testCase,fieldnames(w),{'WD';'WU';'WA';'aM';'W2'});
verifyEqual(testCase,w.WD,1,'AbsTol',1e-12);
verifyGreaterThan(testCase,w.W2,0);
verifyGreaterThan(testCase,diagnostic.rho_M,.8);
d=4*e;
[w,~]=build_kf2_acceptance_weights(d,C2,e,RE,ref,cfg.kf2_acceptance);
verifyLessThanOrEqual(testCase,w.W2*sqrt(mean(d.^2)),cfg.kf2_acceptance.max_rms_recovery_fraction*sqrt(mean(e.^2))+1e-12);
verifyLessThanOrEqual(testCase,w.W2*(max(d)-min(d)),cfg.kf2_acceptance.max_range_recovery_fraction*(max(e)-min(e))+1e-12);
end
