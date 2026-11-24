#!/usr/bin/env python3
"""Reproducible QXU6028 G21 regular-data analysis (L1-L3).

Implements the supplied QXU6028 Virtual Lab Practical Methods Guide.
Run from this directory: python reproduce.py
"""
from pathlib import Path
import json, hashlib, platform, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)

KB = 8.617333e-5       # eV K^-1
NA = 3.816e15          # cm^-2 per atom DoS unit
E_CHARGE = 1.602176634e-19
EMAX = 2.5
BOOTSTRAP_SEED = 20261123
NOISE_SEED = 20261123
N_BOOT = 200
N_NOISE = 500
L3_THRESHOLD = 0.030   # confirmed G21 Session 3 wafer specification: 3.0% of reference conductivity


def fermi(E, EF, T):
    x = np.clip((E - EF) / (KB*T), -60, 60)
    return 1.0 / (np.exp(x) + 1.0)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()


def l1_analysis():
    df = pd.read_csv(DATA / "L1_data.csv")
    par = json.loads((DATA / "L1_params.json").read_text())
    c, T = float(par["c"]), float(par["T"])
    efop, rho = float(par["EF_op_eV"]), float(par["rho_ohm_sq"])
    E = np.linspace(0.0, EMAX, 20001)

    def n_band(ef):
        return NA * np.trapezoid(c*E*fermi(E, ef, T), E)

    def n_fermi(ef):
        # Guide's closed form for ideal linear DoS, also checked against numerical integral.
        kT = KB*T
        return NA*c*kT*(ef*np.log(2.0) + (np.pi**2/12.0)*kT)

    ef = df["EF_eV"].to_numpy(float)
    nb = df["n_band_cm2"].to_numpy(float)
    nf = np.array([n_fermi(x) for x in ef])
    C = nb/nf
    nbop = n_band(efop)
    nfop = n_fermi(efop)
    Cop = nbop/nfop
    muB = 1.0/(E_CHARGE*nbop*rho)
    muF = 1.0/(E_CHARGE*nfop*rho)

    # Independent implementation gates from README_FIRST.txt / Methods Guide.
    checks = {}
    for x, target in [(0.0,1.0),(0.1,2.60),(0.2,5.10)]:
        checks[str(x)] = n_band(x)/n_fermi(x)
        assert abs(checks[str(x)]-target) < 0.06, (x, checks[str(x)], target)
    # The 0.073 eV ~2.00 check is included as a useful extra gate.
    checks["0.073"] = n_band(0.073)/n_fermi(0.073)
    assert abs(checks["0.073"]-2.0) < 0.06
    assert np.all(np.diff(C)>0), "The supplied positive-EF correction factor should be monotonic."

    out = pd.DataFrame({"EF_eV":ef,"n_band_cm2":nb,"n_fermi_cm2":nf,"correction_factor_C":C})
    out.to_csv(TAB/"L1_sweep.csv", index=False)
    op = pd.DataFrame([{
        "EF_op_eV":efop,"T_K":T,"rho_ohm_sq":rho,"n_band_op_cm2":nbop,
        "n_fermi_op_cm2":nfop,"correction_factor_C":Cop,
        "mobility_band_cm2_Vs":muB,"mobility_corrected_cm2_Vs":muF,
        "mobility_ratio_muF_over_muB":muF/muB
    }])
    op.to_csv(TAB/"L1_operating_point.csv", index=False)

    c2=float(np.interp(2.0,C,ef)); c4=float(np.interp(4.0,C,ef))
    pd.DataFrame([{"C2_crossing_EF_eV":c2,"C4_crossing_EF_eV":c4}]).to_csv(TAB/"L1_crossings.csv",index=False)
    plt.figure(figsize=(8,5))
    plt.plot(ef,C,"o-",label="$C(E_F)=n_B/n_F$")
    plt.axvline(efop,ls="--",label=f"Operating $E_F$ = {efop:.4f} eV")
    plt.axvline(c2,ls=":",color="grey",label=f"C=2 at {c2:.3f} eV")
    plt.axvline(c4,ls="-.",color="grey",label=f"C=4 at {c4:.3f} eV")
    plt.axhline(2,color="grey",ls=":",alpha=.6)
    plt.axhline(4,color="grey",ls="-.",alpha=.6)
    plt.xlabel("Fermi energy, $E_F$ (eV)"); plt.ylabel("Correction factor, C")
    plt.title("L1: Carrier-counting correction across the supplied sweep")
    plt.legend(); savefig("L1_correction_factor.png")

    # Composite report figure: both carrier populations and their ratio, generated
    # directly from the untouched G21 inputs during the one-command reproduction.
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].plot(ef, nb/1e12, "o-", label=r"$n_B$ (band-edge count)")
    axs[0].plot(ef, nf/1e12, "s-", label=r"$n_F$ (Fermi-referenced count)")
    axs[0].axvline(efop, ls="--", label=fr"Operating $E_F$ = {efop:.4f} eV")
    axs[0].set_ylabel(r"Carrier density ($10^{12}$ cm$^{-2}$)")
    axs[0].set_title("L1: Carrier-counting conventions across the supplied sweep")
    axs[0].legend(ncol=2)
    axs[1].plot(ef, C, "o-", label=r"$C=n_B/n_F$")
    axs[1].axhline(2, ls=":", label="C = 2")
    axs[1].axhline(4, ls="-.", label="C = 4")
    axs[1].axvline(efop, ls="--", label="Operating point")
    axs[1].set_xlabel(r"Fermi energy, $E_F$ (eV)")
    axs[1].set_ylabel("Correction factor, C")
    axs[1].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG/"L1_carrier_counts_and_C.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Verify numerical integration and closed-form nF at the operating point.
    E2=np.linspace(efop,EMAX,20001)
    nf_num=NA*np.trapezoid(c*E2*fermi(E2,efop,T),E2)
    result={"correction_factor":float(Cop),"n_fermi_op_cm2":float(nfop),"C2_crossing_EF_eV":c2,"C4_crossing_EF_eV":c4,
            "mobility_corrected_cm2_Vs":float(muF),
            "n_band_op_cm2":float(nbop),"mobility_band_cm2_Vs":float(muB),
            "reference_checks":checks,"n_fermi_closed_vs_numeric_relative_error":float(abs(nfop-nf_num)/nf_num)}
    return result


def l2_analysis():
    df = pd.read_csv(DATA/"L2_data.csv")
    meta = json.loads((DATA/"L2_meta.json").read_text())
    T=float(meta["T_K"]); kT=KB*T
    # Shared DoS coefficient from the supplied G21 L1 parameter file; it cancels after separate normalisation.
    c=float(json.loads((DATA/"L1_params.json").read_text())["c"])
    grouped=df.groupby("measurement_index",sort=True)
    means=grouped[["R_H_raw","R_xx_raw"]].mean().reset_index()
    stds=grouped[["R_H_raw","R_xx_raw"]].std(ddof=1).reset_index()
    reps=[g.sort_values("repeat")[["R_H_raw","R_xx_raw"]].to_numpy(float) for _,g in grouped]
    assert len(reps)==43 and all(r.shape==(3,2) for r in reps)
    idx=means.measurement_index.to_numpy(int)
    RH=means.R_H_raw.to_numpy(float); Rs=means.R_xx_raw.to_numpy(float)
    RHn=RH/np.max(np.abs(RH)); Rsn=Rs/np.max(Rs)
    means["R_H_norm"]=RHn; means["R_xx_norm"]=Rsn
    means["R_H_repeat_sd"]=stds.R_H_raw.to_numpy(float)
    means["R_xx_repeat_sd"]=stds.R_xx_raw.to_numpy(float)
    means.to_csv(TAB/"L2_repeat_means.csv",index=False)

    # Data sign-change bracket: consecutive mean Hall values with opposite signs.
    signs=np.sign(RH)
    brackets=[]
    for i in range(len(signs)-1):
        if signs[i]*signs[i+1] < 0:
            brackets.append([int(idx[i]),int(idx[i+1])])

    E=np.linspace(-2.5,2.5,4000)
    EFgrid=np.linspace(-8*kT,9*kT,400)
    def dos(model,Eg):
        if model=="cone": return c*np.abs(E)
        D=Eg/2.0
        return np.where(np.abs(E)>D,c*(np.abs(E)-D),0.0)
    def build_curve(model,Eg):
        g=dos(model,Eg)
        F=fermi(E[None,:],EFgrid[:,None],T)
        n=NA*np.trapezoid(np.where(E[None,:]>=EFgrid[:,None],g[None,:]*F,0.0),E,axis=1)
        p=NA*np.trapezoid(np.where(E[None,:]<=EFgrid[:,None],g[None,:]*(1.0-F),0.0),E,axis=1)
        total=n+p
        rh=(p-n)/(total**2)  # e=1, mu=1; scale cancels after separate normalisation
        rs=1.0/total
        rhn=rh/np.max(np.abs(rh))
        rsn=rs/np.max(rs)
        return np.column_stack([rhn,rsn])

    # Precompute model library once; nearest point fit in normalised 2D space.
    library={"cone":build_curve("cone",0.0)}
    gaps=np.round(np.arange(0.5,8.0001,0.25),2)
    for gap in gaps:
        library[f"gap_{gap:.2f}"]=build_curve("gap",gap*kT)

    def fit_xy(xy):
        results={}
        for name,curve in library.items():
            d2=((xy[:,None,:]-curve[None,:,:])**2).sum(axis=2)
            near=np.argmin(d2,axis=1)
            res=float(np.sqrt(np.mean(np.min(d2,axis=1))))
            results[name]=(res,near,EFgrid[near])
        cone_res=results["cone"][0]
        gap_names=[f"gap_{g:.2f}" for g in gaps]
        best_gap=min(gap_names,key=lambda z:results[z][0])
        gap_res=results[best_gap][0]
        best_gap_kT=float(best_gap.split("_")[1])
        select_gap=(gap_res < 0.85*cone_res and best_gap_kT >= 1.0)
        if select_gap:
            chosen=best_gap; model="hard_gap"; Eg_kT=best_gap_kT
        else:
            chosen="cone"; model="cone"; Eg_kT=0.0
        res,near,efassigned=results[chosen]
        return {"model":model,"chosen":chosen,"Eg_kT":Eg_kT,"residual":res,
                "r_cone":cone_res,"r_best_gap":gap_res,"best_gap_candidate_kT":best_gap_kT,
                "EF":efassigned,"near":near,"all":results}

    central=fit_xy(np.column_stack([RHn,Rsn]))
    assert len(central["EF"])==len(idx)

    # Bootstrap: choose one of three original repeat rows for each measurement index.
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    boot_Eg=[]; boot_models=[]; boot_EF=[]; boot_rcone=[]; boot_rgap=[]
    for b in range(N_BOOT):
        chosen_rows=np.array([r[rng.integers(0,3)] for r in reps])
        rhb=chosen_rows[:,0]; rsb=chosen_rows[:,1]
        xyb=np.column_stack([rhb/np.max(np.abs(rhb)),rsb/np.max(rsb)])
        fit=fit_xy(xyb)
        boot_Eg.append(fit["Eg_kT"]); boot_models.append(fit["model"])
        boot_EF.append(fit["EF"]); boot_rcone.append(fit["r_cone"]); boot_rgap.append(fit["r_best_gap"])
    boot_Eg=np.asarray(boot_Eg); boot_EF=np.asarray(boot_EF)
    eg_p16,eg_med,eg_p84=np.percentile(boot_Eg,[16,50,84])
    ef_p16,ef_med,ef_p84=np.percentile(boot_EF,[16,50,84],axis=0)
    gap_fraction=float(np.mean(np.array(boot_models)=="hard_gap"))
    eg_unc=float((eg_p84-eg_p16)/2.0)

    # Save trajectory and bootstrap bands.
    traj=pd.DataFrame({"measurement_index":idx,"EF_central_eV":central["EF"],
                       "EF_p16_eV":ef_p16,"EF_median_eV":ef_med,"EF_p84_eV":ef_p84})
    traj.to_csv(TAB/"L2_EF_trajectory.csv",index=False)
    pd.DataFrame({"bootstrap_Eg_kT":boot_Eg,"bootstrap_model":boot_models,
                  "r_cone":boot_rcone,"r_best_gap":boot_rgap}).to_csv(TAB/"L2_bootstrap.csv",index=False)
    pd.DataFrame([{"central_model":central["model"],"central_Eg_kT":central["Eg_kT"],
                   "r_cone":central["r_cone"],"r_best_gap":central["r_best_gap"],
                   "best_gap_candidate_kT":central["best_gap_candidate_kT"],
                   "bootstrap_Eg_median_kT":eg_med,"bootstrap_Eg_p16_kT":eg_p16,
                   "bootstrap_Eg_p84_kT":eg_p84,"gap_selection_fraction":gap_fraction,
                   "bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_replicates":N_BOOT}]).to_csv(TAB/"L2_model_selection.csv",index=False)

    # Raw three-repeat plot in original units, preserving signs and acquisition order.
    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for rep in [1,2,3]:
        sub=df[df.repeat==rep]
        axs[0].plot(sub.measurement_index,sub.R_H_raw*1e12,".-",alpha=.7,label=f"Repeat {rep}")
        axs[1].plot(sub.measurement_index,sub.R_xx_raw*1e12,".-",alpha=.7,label=f"Repeat {rep}")
    axs[0].plot(idx,RH*1e12,"ko-",lw=1.8,label="Repeat mean")
    axs[1].plot(idx,Rs*1e12,"ko-",lw=1.8,label="Repeat mean")
    for a,b in brackets:
        axs[0].axvspan(a,b,alpha=.15,color="red")
        axs[1].axvspan(a,b,alpha=.15,color="red")
    axs[0].axhline(0,color="black",lw=.8)
    axs[0].set_ylabel("$R_H$ (10$^{-12}$ raw units)")
    axs[1].set_ylabel("$R_{xx}$ (10$^{-12}$ raw units)")
    axs[1].set_xlabel("Measurement index")
    axs[0].set_title("L2: Three raw repeats and mean values (red band brackets Hall sign change)")
    axs[0].legend(ncol=4); axs[1].legend(ncol=4)
    fig.tight_layout(); fig.savefig(FIG/"L2_raw_repeats.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    # Explicit residual comparison for the cone and best hard-gap candidate.
    plt.figure(figsize=(6,4))
    plt.bar(["Cone",f"Best gap ({central['best_gap_candidate_kT']:.2f} kBT)"],
            [central["r_cone"],central["r_best_gap"]])
    plt.ylabel("Nearest-curve RMS residual")
    plt.title("L2: Residual comparison before the 15% selection gate")
    savefig("L2_residual_comparison.png")

    # Mean measured points and selected curve.
    curve=library[central["chosen"]]
    # Display-only crop: show the model curve over the EF interval assigned to
    # measured points. The 400-point fit grid and all residuals are unchanged.
    ef_plot_min=float(np.min(central["EF"]))
    ef_plot_max=float(np.max(central["EF"]))
    plot_mask=(EFgrid >= ef_plot_min) & (EFgrid <= ef_plot_max)
    curve_plot=curve[plot_mask]
    plt.figure(figsize=(7,6))
    plt.plot(curve_plot[:,0],curve_plot[:,1],"-",lw=2,label=f"Selected {central['model']} model (measured EF span)")
    plt.scatter(RHn,Rsn,c=idx,cmap="viridis",s=32,label="Measured means")
    plt.colorbar(label="Measurement index")
    plt.xlabel("Normalised signed $R_H$"); plt.ylabel("Normalised $R_{xx}$")
    plt.title("L2: Nearest-curve fit in normalised pair space")
    plt.legend(); savefig("L2_model_fit.png")

    plt.figure(figsize=(8,5))
    plt.fill_between(idx,ef_p16*1000,ef_p84*1000,alpha=.25,label="Bootstrap 16–84% interval")
    plt.plot(idx,central["EF"]*1000,"o-",label="Central-fit $E_F$")
    plt.xlabel("Measurement index"); plt.ylabel("Assigned Fermi level (meV)")
    plt.title("L2: Fermi-level trajectory with bootstrap uncertainty")
    plt.legend(); savefig("L2_EF_trajectory.png")

    # Composite report figure: separate signed Hall and sheet-resistance overlays.
    # The measured x-coordinate is the nearest-curve assigned EF; uncertainty bars
    # are the repeat SD, scaled by the same independent normalisers as the means.
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ef_model_meV = EFgrid*1000
    ef_data_meV = central["EF"]*1000
    axs[0].plot(ef_model_meV, curve[:,0], lw=2, label=f"{central['model']} model prediction")
    axs[0].errorbar(ef_data_meV, RHn, yerr=stds.R_H_raw.to_numpy(float)/np.max(np.abs(RH)),
                    fmt="o", capsize=2, label="Measured repeat mean ± SD")
    axs[0].axhline(0, color="black", lw=.8)
    axs[0].set_ylabel(r"Normalised signed $R_H$")
    axs[0].legend()
    axs[1].plot(ef_model_meV, curve[:,1], lw=2, label=f"{central['model']} model prediction")
    axs[1].errorbar(ef_data_meV, Rsn, yerr=stds.R_xx_raw.to_numpy(float)/np.max(Rs),
                    fmt="o", capsize=2, label="Measured repeat mean ± SD")
    axs[1].set_xlabel(r"Assigned Fermi level, $E_F$ (meV)")
    axs[1].set_ylabel(r"Normalised $R_{xx}$")
    axs[1].legend()
    fig.suptitle("L2: Separate Hall and sheet-resistance overlays")
    fig.tight_layout()
    fig.savefig(FIG/"L2_observable_overlays.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    l2res={"model":central["model"],"Eg_kT":float(central["Eg_kT"]),
           # Store the required 16th–84th percentile interval, not a scalar half-width.
           "Eg_uncertainty_kT":[float(eg_p16),float(eg_p84)],
           "EF_trajectory_eV":[float(x) for x in central["EF"]],
           "r_cone":float(central["r_cone"]),"r_best_gap":float(central["r_best_gap"]),
           "best_gap_candidate_kT":float(central["best_gap_candidate_kT"]),
           "bootstrap_Eg_median_kT":float(eg_med),"bootstrap_Eg_p16_kT":float(eg_p16),
           "bootstrap_Eg_p84_kT":float(eg_p84),"gap_selection_fraction":gap_fraction,
           "EF_p16_eV":[float(x) for x in ef_p16],"EF_p84_eV":[float(x) for x in ef_p84],
           "sign_change_brackets":brackets,"bootstrap_seed":BOOTSTRAP_SEED,
           "bootstrap_replicates":N_BOOT}
    return l2res


def l3_analysis():
    A=np.load(DATA/"L3_A.npy")
    d_reps=np.loadtxt(DATA/"L3_d.csv",delimiter=",")
    assert A.shape==(128,100),A.shape
    assert d_reps.shape==(3,128),d_reps.shape
    row_sum_max=float(np.max(np.abs(A.sum(axis=1))))
    assert row_sum_max<1e-11,row_sum_max
    dbar=d_reps.mean(axis=0)
    npx=10; N=100
    # First-difference matrix, one row per horizontal/vertical adjacent pair.
    rows=[]
    for i in range(npx):
        for j in range(npx):
            k=i*npx+j
            if j<npx-1:
                r=np.zeros(N); r[k]=-1; r[k+1]=1; rows.append(r)
            if i<npx-1:
                r=np.zeros(N); r[k]=-1; r[k+npx]=1; rows.append(r)
    L=np.asarray(rows)
    one=np.ones((N,1))/np.sqrt(N)
    P=np.eye(N)-one@one.T
    eta0=1e-2*np.linalg.norm(A,2)**2/np.linalg.norm(L,2)**2

    def solver(eta):
        M=P@(A.T@A+eta*(L.T@L))@P+1e-9*np.eye(N)
        return M
    def reconstruct_many(D,eta=eta0):
        # D is (n_vectors, 128). Solve the exact guide linear system for all RHS at once.
        M=solver(eta)
        B=P@(A.T@D.T)
        X=np.linalg.lstsq(M,B,rcond=None)[0]
        X=X-X.mean(axis=0,keepdims=True)
        return X.T
    mhat=reconstruct_many(dbar[None,:],eta0)[0]
    m2=mhat.reshape(10,10)
    assert abs(float(np.mean(mhat)))<1e-12
    assert np.all(np.isfinite(mhat))

    # Regularisation sensitivity.
    eta_factors=np.array([0.01,0.1,1,10,100],float)
    sens=[]
    for fac in eta_factors:
        mm=reconstruct_many(dbar[None,:],eta0*fac)[0]
        sens.append({"factor":fac,"eta":eta0*fac,
                     "residual_norm":float(np.linalg.norm(A@mm-dbar)),
                     "roughness_norm":float(np.linalg.norm(L@mm)),
                     "max_abs_map":float(np.max(np.abs(mm)))})
    pd.DataFrame(sens).to_csv(TAB/"L3_eta_sensitivity.csv",index=False)

    # Features: connected components of pixels above 50% of each sign's global extremum.
    coords=np.linspace(-2.7,2.7,10)
    def components(mask):
        seen=set(); comps=[]
        for r in range(10):
            for c0 in range(10):
                if not mask[r,c0] or (r,c0) in seen: continue
                stack=[(r,c0)]; seen.add((r,c0)); comp=[]
                while stack:
                    x,y=stack.pop(); comp.append((x,y))
                    for dx in (-1,0,1):
                        for dy in (-1,0,1):
                            nx,ny=x+dx,y+dy
                            if 0<=nx<10 and 0<=ny<10 and mask[nx,ny] and (nx,ny) not in seen:
                                seen.add((nx,ny)); stack.append((nx,ny))
                comps.append(comp)
        return comps
    features=[]
    for sign,label in [(1,"positive"),(-1,"negative")]:
        peak=float(np.max(m2) if sign==1 else np.min(m2))
        threshold=0.5*abs(peak)
        mask=(m2>=threshold) if sign==1 else (m2<=-threshold)
        for comp in components(mask):
            vals=np.array([m2[r,c0] for r,c0 in comp])
            # Feature coordinate is magnitude-weighted centroid; peak pixel is retained too.
            weights=np.abs(vals)
            rr=np.array([p[0] for p in comp]); cc=np.array([p[1] for p in comp])
            cx=float(np.sum(coords[cc]*weights)/np.sum(weights))
            cy=float(np.sum(coords[rr]*weights)/np.sum(weights))
            ip=int(np.argmax(vals) if sign==1 else np.argmin(vals))
            features.append({"sign":label,"peak_fraction":float(vals[ip]),
                             "peak_x_probe_pitch":float(coords[cc[ip]]),
                             "peak_y_probe_pitch":float(coords[rr[ip]]),
                             "centroid_x_probe_pitch":cx,"centroid_y_probe_pitch":cy,
                             "pixels_merged":len(comp)})
    features=sorted(features,key=lambda z:abs(z["peak_fraction"]),reverse=True)

    # Resolution test: central pixel and off-centre pixel; weighted positive-lobe FWHM.
    def point_spread_fwhm(row,col):
        q=np.zeros(N); q[row*npx+col]=1.0; q-=q.mean()
        qhat=reconstruct_many((A@q)[None,:],eta0)[0].reshape(10,10)
        w=np.maximum(qhat,0.0)
        X,Y=np.meshgrid(coords,coords)
        r2=(X-coords[col])**2+(Y-coords[row])**2
        sig2=np.sum(w*r2)/(2*np.sum(w))
        return 2.355*np.sqrt(sig2),qhat
    fwhm_c,psf_c=point_spread_fwhm(4,4)
    fwhm_o,psf_o=point_spread_fwhm(2,7)
    resolution=max(fwhm_c,fwhm_o)

    # Noise-only reconstruction, 500 vectors, channel-specific repeat SD.
    sigma_j=np.std(d_reps,axis=0,ddof=1)
    rng=np.random.default_rng(NOISE_SEED)
    noise=rng.normal(0.0,sigma_j,size=(N_NOISE,128))
    noise_maps=reconstruct_many(noise,eta0)
    noise_max=np.max(np.abs(noise_maps),axis=1)
    delta95=float(np.percentile(noise_max,95))
    a=float(np.max(np.abs(mhat)))
    lo=max(0.0,a-delta95); hi=a+delta95
    if hi<L3_THRESHOLD:
        confidence="clear_pass"; passed=True
    elif lo>L3_THRESHOLD:
        confidence="clear_fail"; passed=False
    else:
        confidence="borderline"; passed=False

    # L3 figures.
    plt.figure(figsize=(7,6))
    im=plt.imshow(m2,origin="lower",extent=[-3,3,-3,3],aspect="equal",cmap="coolwarm")
    plt.colorbar(im,label="Relative conductivity deviation, $m$ (fraction)")
    plt.xlabel("x (probe pitches)"); plt.ylabel("y (probe pitches)")
    plt.title("L3: Reconstructed zero-mean relative conductivity map")
    savefig("L3_conductivity_map.png")

    plt.figure(figsize=(7,5))
    plt.plot([x["roughness_norm"] for x in sens],[x["residual_norm"] for x in sens],"o-")
    for x in sens: plt.annotate(f"×{x['factor']:g}",(x["roughness_norm"],x["residual_norm"]))
    plt.xlabel("Roughness norm ||Lm||"); plt.ylabel("Data residual ||Am − d̄||")
    plt.title("L3: Regularisation sensitivity")
    savefig("L3_eta_sensitivity.png")

    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,psf,title in [(axs[0],psf_c,"Central injection"),(axs[1],psf_o,"Off-centre injection")]:
        im=ax.imshow(psf,origin="lower",extent=[-3,3,-3,3],cmap="viridis")
        ax.set_title(title); ax.set_xlabel("x (probe pitches)"); ax.set_ylabel("y (probe pitches)")
        fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle(f"L3: Point-spread tests (FWHM central={fwhm_c:.3f}s, off-centre={fwhm_o:.3f}s)")
    fig.tight_layout(); fig.savefig(FIG/"L3_point_spread.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    plt.figure(figsize=(7,5))
    plt.hist(noise_max,bins=30)
    plt.axvline(delta95,ls="--",label=f"95th percentile = {delta95:.5f}")
    plt.xlabel("max |noise-only reconstructed map|"); plt.ylabel("Count")
    plt.title("L3: 500 noise-only reconstructions")
    plt.legend(); savefig("L3_noise_detection_limit.png")

    pd.DataFrame({"noise_max_abs_map":noise_max}).to_csv(TAB/"L3_noise_reconstructions.csv",index=False)
    pd.DataFrame(m2,index=coords,columns=coords).to_csv(TAB/"L3_map_10x10.csv",index_label="y_x_probe_pitch")
    pd.DataFrame(features).to_csv(TAB/"L3_features.csv",index=False)
    pd.DataFrame([{"eta0":eta0,"eta_factor_used":1.0,"a_max_abs_map":a,
                   "delta95":delta95,"decision_lower":lo,"decision_upper":hi,
                   "threshold":L3_THRESHOLD,"confidence":confidence,"pass":passed,
                   "resolution_central_probe_pitch":fwhm_c,
                   "resolution_offcentre_probe_pitch":fwhm_o,
                   "resolution_conservative_probe_pitch":resolution,
                   "noise_seed":NOISE_SEED,"noise_vectors":N_NOISE,
                   "repeat_noise_mean_sd":float(np.mean(sigma_j)),
                   "row_sum_max_abs":row_sum_max}]).to_csv(TAB/"L3_summary.csv",index=False)

    l3res={"map":[float(x) for x in mhat],"features":features,
           "resolution_probe_pitch":float(resolution),
           "detection_limit_fraction":delta95,"pass":bool(passed),"confidence":confidence,
           "eta0":float(eta0),"eta_factor_used":1.0,"max_abs_map":a,
           "decision_interval_fraction":[float(lo),float(hi)],
           "threshold_fraction":L3_THRESHOLD,
           "resolution_central_probe_pitch":float(fwhm_c),
           "resolution_offcentre_probe_pitch":float(fwhm_o),
           "noise_seed":NOISE_SEED,"noise_vectors":N_NOISE,
           "repeat_noise_sd_mean":float(np.mean(sigma_j)),"row_sum_max_abs":row_sum_max,
           "eta_sensitivity":sens}
    return l3res


def main():
    # Verify supplied raw data against the group's manifest before analysis.
    manifest=json.loads((DATA/"manifest.json").read_text()) if (DATA/"manifest.json").exists() else None
    manifest_checks=[]
    if manifest:
        for item in manifest["files"]:
            raw=(DATA/item["name"]).read_bytes()
            actual=hashlib.sha256(raw).hexdigest()
            ok=(len(raw)==item["bytes"] and actual==item["sha256"])
            manifest_checks.append({"file":item["name"],"bytes_match":len(raw)==item["bytes"],"sha256_match":actual==item["sha256"]})
            assert ok, f"Manifest mismatch for {item['name']}"
    (ROOT/"manifest_verification.json").write_text(json.dumps(manifest_checks,indent=2)+"\n")
    l1=l1_analysis(); l2=l2_analysis(); l3=l3_analysis()
    # Assessment-critical shape, finiteness and numerical checks.
    n_unique=int(pd.read_csv(DATA/"L2_data.csv")["measurement_index"].nunique())
    assert len(l2["EF_trajectory_eV"]) == n_unique, (len(l2["EF_trajectory_eV"]), n_unique)
    assert len(l3["map"]) == 100 and np.all(np.isfinite(l3["map"]))
    assert l1["n_fermi_closed_vs_numeric_relative_error"] < 1e-5
    required_report_figures=["L1_carrier_counts_and_C.png", "L2_observable_overlays.png"]
    assert all((FIG/name).is_file() for name in required_report_figures)
    template_path=DATA/"results_template.json"
    template=json.loads(template_path.read_text())
    # Fill the official supplied schema; do not invent or rename fields.
    result=template.copy()
    commit_file=ROOT/"pipeline_commit.txt"
    commit_hash=commit_file.read_text().strip() if commit_file.exists() else "UNCOMMITTED"
    result["pipeline_commit"]=commit_hash
    result["L1"].update({k:l1[k] for k in ["correction_factor","n_fermi_op_cm2","mobility_corrected_cm2_Vs"]})
    result["L2"].update({k:l2[k] for k in ["model","Eg_kT","Eg_uncertainty_kT","EF_trajectory_eV"]})
    result["L3"].update({k:l3[k] for k in ["map","features","resolution_probe_pitch","detection_limit_fraction","pass","confidence"]})
    assert set(result)==set(template), "Top-level results.json keys differ from official template"
    for section in ("L1","L2","L3"):
        assert set(result[section])==set(template[section]), f"{section} keys differ from official template"
    (ROOT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    detailed={"group":"G21","dataset_set":"regular","L1":l1,"L2":l2,"L3":l3,
              "constants":{"kB_eV_K":KB,"N_A_cm2":NA,"elementary_charge_C":E_CHARGE},
              "seeds":{"L2_bootstrap":BOOTSTRAP_SEED,"L3_noise":NOISE_SEED},
              "manifest_verification":manifest_checks,
              "notes":["L3 threshold 0.030 (3.0% of reference conductivity) is confirmed from the G21 Session 3 wafer-specification slide.",
                       "L2 DoS coefficient c was read from the group's L1_params.json; separate normalisation removes its fixed scale.",
                       "The exact template results.json is preserved; detailed diagnostics are in tables/ and analysis_summary.json."]}
    (ROOT/"analysis_summary.json").write_text(json.dumps(detailed,indent=2)+"\n")
    # concise human-readable notes
    # Deterministic human-readable run record. This file is regenerated with the
    # numerical outputs, so it cannot retain an obsolete provisional threshold.
    commit_hash=(ROOT/"pipeline_commit.txt").read_text().strip() if (ROOT/"pipeline_commit.txt").exists() else "UNCOMMITTED"
    generated_figures=sorted(x.name for x in FIG.glob("*.png"))
    txt=f"""QXU6028 Polymer Devices — G21 regular analysis
Dataset: G21 regular only
Manifest verification: PASS ({len(manifest_checks)} supplied files)
Pipeline code commit: {commit_hash}

L1
- DoS scale coefficient c: {float(json.loads((DATA/'L1_params.json').read_text())['c']):.10f}
- Exact operating EF: {float(json.loads((DATA/'L1_params.json').read_text())['EF_op_eV']):.9f} eV
- Temperature: {float(json.loads((DATA/'L1_params.json').read_text())['T']):.1f} K
- Sheet resistivity: {float(json.loads((DATA/'L1_params.json').read_text())['rho_ohm_sq']):.6f} ohm/square
- n_band(op): {l1['n_band_op_cm2']:.6e} cm^-2
- n_fermi(op): {l1['n_fermi_op_cm2']:.6e} cm^-2
- Correction factor C: {l1['correction_factor']:.6f}
- Corrected mobility: {l1['mobility_corrected_cm2_Vs']:.3f} cm^2 V^-1 s^-1
- Closed-form vs numerical nF relative difference: {l1['n_fermi_closed_vs_numeric_relative_error']:.3e}
- Reference checks: {json.dumps(l1['reference_checks'], sort_keys=True)}

L2
- Unique measurement indices: {n_unique}; trajectory length: {len(l2['EF_trajectory_eV'])}
- Central model: {l2['model']}; Eg/kBT: {l2['Eg_kT']:.3f}
- Cone residual: {l2['r_cone']:.6f}; best-gap residual: {l2['r_best_gap']:.6f}
- Bootstrap Eg/kBT median [16th,84th]: {l2['bootstrap_Eg_median_kT']:.3f} [{l2['bootstrap_Eg_p16_kT']:.3f}, {l2['bootstrap_Eg_p84_kT']:.3f}]
- Gap-selection fraction: {l2['gap_selection_fraction']:.3f}; bootstrap runs: {N_BOOT}; seed: {BOOTSTRAP_SEED}
- Hall sign-change bracket(s): {l2['sign_change_brackets']}

L3
- eta0: {l3['eta0']:.6e}
- max|m|: {l3['max_abs_map']:.6f} ({100*l3['max_abs_map']:.4f}%)
- Noise detection limit delta95: {l3['detection_limit_fraction']:.8f} fraction ({100*l3['detection_limit_fraction']:.5f} percentage points)
- Decision interval: [{100*l3['decision_interval_fraction'][0]:.4f}%, {100*l3['decision_interval_fraction'][1]:.4f}%]
- Confirmed G21 Session 3 threshold: {100*l3['threshold_fraction']:.2f}%
- Decision: {l3['confidence']} (pass flag={l3['pass']})
- Conservative spatial resolution: {l3['resolution_probe_pitch']:.3f} probe pitches
- Noise simulations: {N_NOISE}; seed: {NOISE_SEED}

Generated figures ({len(generated_figures)}):
""" + "".join(f"- {name}\n" for name in generated_figures)
    (ROOT/"RESULTS_README.txt").write_text(txt)
    (ROOT/"run_log.txt").write_text(txt)
    print(txt)
    print("L1 checks:",json.dumps(l1["reference_checks"],indent=2))
    print("Output files written to",ROOT)

if __name__=="__main__": main()