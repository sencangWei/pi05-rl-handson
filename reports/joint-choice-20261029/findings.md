# Findings

Absolute pilot: train 2 controls then J4 upper-limit rejection; holdout 0 controls then 1.424rad J7 jump rejection; 0/2 success. J7 train coverage is narrow.
Current controller: absolute position PD, kp150, kd2sqrt150, control20Hz. Existing pairs are fresh physically executed reconstructed candidate joint targets, not original historical human commands.
Velocity has no validated matched controller or desired-velocity labels; measured qvel is not a velocity command. Excluded from empirical winner claim.
Residual encoding uses one pre-action observed joint anchor for all future10 targets. Gripper remains absolute. Raw pairs remain unchanged.
State normalization kept identical; residual action statistics must reflect transformed windows from train only, not absolute action statistics. This normalization difference is part of representation experiment.

Residual expert replay passed both sources, total965controls; max physics error4.16e-11. Real residual training completed200updates, exact baseline sampled-window order,115.85s including save. Residual train state rejected firstJ4target at -0.0692007063rad (upper-0.0698rad). Holdout ran600controls, no native goal and no negative/open gripper command; final100controls allgrasp=true. Both models0/2. Residual avoided initial J7snap but does not prove a functional policy. Recorded controller labels remain absolute; residual is a reversible training representation. No UI promotion. Own comparison server stopped after evaluation.
