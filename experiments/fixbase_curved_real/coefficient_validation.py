"""Train/hold-out test of empirical force gains; not a new aerodynamic model.

All gains use 10--12 Hz only. 13--15 Hz are held out. The existing yaw is
frozen, and the earlier shared per-record phase is retained for every gain.
"""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_comparison import write_csv
from rect_compare import (OUTPUT as PREVIOUS, AXES, harmonic_fit, waveform,
                          evaluate, yaw_rotate, NPHASE)

FOLDER=Path(__file__).resolve().parent
OUTPUT=FOLDER/'coefficient_validation'


def gain_fit(prediction, measured):
    """Nonnegative scalar, zero intercept, ordinary least squares."""
    x=np.asarray(prediction).ravel();y=np.asarray(measured).ravel()
    den=x@x
    if den<1e-20:
        return None
    return max(0.,float((x@y)/den))


def force_scores(pred,real):
    error=pred-real
    rms=np.sqrt(np.mean(error**2,axis=0))
    ref=np.sqrt(np.mean((real-real.mean(0))**2,axis=0))
    return {**{f'{a}_rmse_mN':float(rms[i]*1000) for i,a in enumerate(AXES[:3])},
            **{f'{a}_nrmse':float(rms[i]/max(ref[i],1e-12)) for i,a in enumerate(AXES[:3])},
            'vector_nrmse':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))/
                                 np.sqrt(np.mean(np.sum((real-real.mean(0))**2,axis=1))))}


def main():
    OUTPUT.mkdir(exist_ok=True)
    summary=json.loads((PREVIOUS/'comparison_summary.json').read_text())
    rows=summary['lift']
    simulations=json.loads((PREVIOUS/'momentum_summary.json').read_text())
    realruns=json.loads((PREVIOUS/'real_runs.json').read_text())
    smap={(s['kind'],s['nominal_hz']):s for s in simulations}
    gmap={(r['kind'],r['hz']):r for r in rows}
    fits={}
    lift_results=[]
    for scope in ('shared','curve','rect'):
        selected=[r for r in rows if scope=='shared' or r['kind']==scope]
        train=[r for r in selected if r['hz']<=12]
        k=gain_fit([r['sim_lift_mN'] for r in train],[r['real_lift_mN'] for r in train])
        fit=dict(gain=k,training_hz=[10,11,12],held_out_hz=[13,14,15])
        for split in ('train','test'):
            subset=[r for r in selected if (r['hz']<=12)==(split=='train')]
            err=[]
            for r in subset:
                corrected=k*r['sim_lift_mN']
                pct=100*(corrected/r['real_lift_mN']-1)
                lift_results.append(dict(scope=scope,kind=r['kind'],hz=r['hz'],split=split,
                                         gain=k,real_mN=r['real_lift_mN'],raw_sim_mN=r['sim_lift_mN'],
                                         calibrated_mN=corrected,error_pct=pct,
                                         real_to_sim_ratio=r['real_lift_mN']/r['sim_lift_mN']))
                err.append(pct)
            fit[split+'_mape_pct']=float(np.mean(np.abs(err)))
            fit[split+'_max_abs_error_pct']=float(np.max(np.abs(err)))
        fits[scope]=fit
    write_csv(OUTPUT/'lift_holdout.csv',lift_results)
    # Cross-wing transfer uses the already fitted gain, never refits on test data.
    transfer=[]
    for source,dest in [('curve','rect'),('rect','curve')]:
        subset=[r for r in rows if r['kind']==dest and r['hz']>=13]
        errors=[100*(fits[source]['gain']*r['sim_lift_mN']/r['real_lift_mN']-1) for r in subset]
        transfer.append(dict(fit_wing=source,test_wing=dest,gain=fits[source]['gain'],
                             held_out_mape_pct=float(np.mean(np.abs(errors))),errors_pct=errors))
    # Six-axis mean gain diagnostic. Moments require a known common origin and
    # gravity-tare definition before this becomes a physical moment validation.
    mean6=[]
    for kind in ('curve','rect'):
        theta=np.radians(summary['orientation'][kind]['yaw_deg'])
        pred=[];real=[]
        for hz in range(10,16):
            s=smap[kind,hz]
            w=np.array([s['mean_wrenches'][f'aero_{a}_{"N" if a[0]=="F" else "Nm"}'] for a in AXES])
            w=np.r_[yaw_rotate(w[None,:3],theta)[0],yaw_rotate(w[None,3:],theta)[0]]
            rr=[r for r in realruns if r['kind']==kind and r['hz']==hz]
            pred.append(w);real.append([np.mean([r[a+'_mean'] for r in rr]) for a in AXES])
        pred=np.array(pred);real=np.array(real)
        for i,axis in enumerate(AXES):
            coefficient=gain_fit(pred[:3,i],real[:3,i])
            model_force_only=fits[kind]['gain']*pred[3:,i]
            rmse=float(np.sqrt(np.mean((model_force_only-real[3:,i])**2)))
            mean6.append(dict(kind=kind,axis=axis,unit='N' if i<3 else 'Nm',
                              diagnostic_axis_gain=coefficient,
                              lift_gain_held_out_rmse=rmse,
                              measured_held_out_rms=float(np.sqrt(np.mean(real[3:,i]**2))),
                              status='force with candidate yaw' if i<3 else
                              'CONDITIONAL ONLY: assumed coincident origins; gravity tare not reconciled'))
    write_csv(OUTPUT/'six_axis_mean_diagnostic.csv',mean6)
    saved=np.load(PREVIOUS/'harmonic_waveforms.npz')
    pairs=[]
    for r in realruns:
        kind,hz=r['kind'],r['hz'];key=f'{kind}_{hz}_{r["repeat"]}'
        s=smap[kind,hz]
        simwave=saved[key+'_sim'];realwave=saved[key+'_real']
        theta=np.radians(summary['orientation'][kind]['yaw_deg'])
        pred,old=evaluate(dict(sim_wave=simwave,real_wave=realwave),theta)
        shift=round(old['phase_shift_deg']*NPHASE/360)
        data=np.load(PREVIOUS/s['source']);t=data['time_s'];start,end=s['averaging_interval_s']
        keep=(t>=start)&(t<=end)
        af=np.column_stack([data[f'aero_{axis}_N'] for axis in AXES[:3]])
        ac,_=harmonic_fit(t[keep],af[keep],s['measured_wing_frequency_hz'])
        ac[0]=[s['mean_wrenches'][f'aero_{axis}_N'] for axis in AXES[:3]]
        aw=yaw_rotate(np.roll(waveform(ac),shift,axis=0),theta)
        pairs.append(dict(kind=kind,hz=hz,repeat=r['repeat'],pred=pred,real=realwave,aero=aw))
    wavegains={};scores=[];aggregates=[]
    for kind in ('curve','rect'):
        train=[p for p in pairs if p['kind']==kind and p['hz']<=12]
        x=np.concatenate([p['pred'] for p in train]);y=np.concatenate([p['real'] for p in train])
        vector_gain=gain_fit(x,y)
        axis_gains=np.array([gain_fit(x[:,i],y[:,i]) for i in range(3)])
        wavegains[kind]=dict(single_force_gain=vector_gain,axis_gains=dict(zip(AXES[:3],axis_gains.tolist())))
        for pair in [p for p in pairs if p['kind']==kind]:
            k=fits[kind]['gain']
            predictions={
                'original':pair['pred'],
                'lift_gain_whole_force':k*pair['pred'],
                'lift_gain_aero_only_frozen_motion':pair['pred']+(k-1)*pair['aero'],
                'waveform_fitted_scalar':vector_gain*pair['pred'],
                'waveform_fitted_axis_gains':axis_gains*pair['pred'],
            }
            pair['variants']=predictions
            for mode,prediction in predictions.items():
                scores.append(dict(kind=kind,hz=pair['hz'],repeat=pair['repeat'],
                                   split='train' if pair['hz']<=12 else 'test',mode=mode,
                                   **force_scores(prediction,pair['real'])))
        for split in ('train','test'):
            for mode in predictions:
                rr=[r for r in scores if r['kind']==kind and r['mode']==mode and r['split']==split]
                aggregates.append(dict(kind=kind,split=split,mode=mode,
                                       **{key:float(np.mean([r[key] for r in rr])) for key in force_scores(x,y)}))
    write_csv(OUTPUT/'force_waveform_per_run.csv',scores)
    write_csv(OUTPUT/'force_waveform_summary.csv',aggregates)
    result=dict(lift_fits=fits,cross_wing_transfer=transfer,waveform_gains=wavegains,
                waveform_errors=aggregates,moment_validation='NOT ESTABLISHED: origin, gravity baseline and impulsive torque sampling unresolved',
                physical_scope='Empirical output calibration / frozen-motion diagnostics. No modified-fluid dynamics rerun.',
                sources={str(p.relative_to(FOLDER)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [PREVIOUS/'comparison_summary.json',PREVIOUS/'momentum_summary.json',PREVIOUS/'real_runs.json',
                          PREVIOUS/'harmonic_waveforms.npz',Path(__file__)]})
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,kind in zip(axes,('curve','rect')):
        subset=[r for r in rows if r['kind']==kind]
        x=np.array([r['hz'] for r in subset]);raw=np.array([r['sim_lift_mN'] for r in subset])
        ax.axvspan(12.5,15.4,color='#b6ddbc',alpha=.25,label='Held out: 13-15 Hz')
        ax.errorbar(x,[r['real_lift_mN'] for r in subset],yerr=[r['real_sd_mN'] for r in subset],fmt='ko-',capsize=3,label='Measured +/- SD')
        ax.plot(x,raw,'s:',color='gray',label='Original simulation')
        ax.plot(x,raw*fits[kind]['gain'],'o-',color='#1879b2',label=f'Wing-specific k={fits[kind]["gain"]:.3f}')
        ax.plot(x,raw*fits['shared']['gain'],'--',color='#d55e00',label=f'Shared k={fits["shared"]["gain"]:.3f}')
        ax.set(title=kind,xlabel='Nominal frequency [Hz]',ylabel='Mean Fz [mN]',xlim=(9.7,15.3));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUTPUT/'lift_gain_holdout.png',dpi=170);fig.savefig(OUTPUT/'lift_gain_holdout.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ki,kind in enumerate(('curve','rect')):
        pair=next(p for p in pairs if p['kind']==kind and p['hz']==15 and p['repeat']=='01')
        for i,ax in enumerate(axes[ki]):
            phase=np.arange(NPHASE)/NPHASE
            ax.plot(phase,pair['real'][:,i]*1000,'k',label='Measured')
            ax.plot(phase,pair['pred'][:,i]*1000,':',color='gray',label='Original')
            ax.plot(phase,pair['variants']['lift_gain_whole_force'][:,i]*1000,color='#1879b2',label='Mean-lift gain applied')
            ax.plot(phase,pair['variants']['waveform_fitted_axis_gains'][:,i]*1000,'--',color='#d55e00',label='Separate axis gains (diagnostic)')
            ax.set(title=f'{kind}, 15 Hz held out: {AXES[i]}',xlabel='Time-aligned phase',ylabel='Force [mN]');ax.grid(alpha=.2)
            if i==0:ax.legend(fontsize=7)
    fig.savefig(OUTPUT/'force_gain_waveforms.png',dpi=170);plt.close(fig)
    write_report(result,rows,mean6)
    print(json.dumps({'lift':fits,'transfer':transfer,'waveform_gains':wavegains,
                      'test_force_errors':[r for r in aggregates if r['split']=='test']},indent=2))
    return result


def write_report(result,rows,mean6):
    fits=result['lift_fits'];errors=result['waveform_errors'];wg=result['waveform_gains']
    lines=['|系数适用范围|10–12 Hz 拟合 k|13–15 Hz 平均绝对相对误差|最大绝对相对误差|','|---|---:|---:|---:|']
    for key,label in [('curve','曲面翼'),('rect','矩形翼'),('shared','两翼型共用')]:
        f=fits[key];lines.append(f"|{label}|{f['gain']:.4f}|{f['test_mape_pct']:.2f}%|{f['test_max_abs_error_pct']:.2f}%|")
    wave=['|翼型|校正方式|Fx NRMSE|Fy NRMSE|Fz NRMSE|三轴向量 NRMSE|','|---|---|---:|---:|---:|---:|']
    names={'original':'原始','lift_gain_whole_force':'平均升力系数乘整个力',
           'lift_gain_aero_only_frozen_motion':'仅缩放气动力，运动固定',
           'waveform_fitted_scalar':'周期力拟合单系数','waveform_fitted_axis_gains':'逐轴拟合（诊断）'}
    for e in errors:
        if e['split']=='test':
            wave.append(f"|{e['kind']}|{names[e['mode']]}|{e['Fx_nrmse']*100:.1f}%|{e['Fy_nrmse']*100:.1f}%|{e['Fz_nrmse']*100:.1f}%|{e['vector_nrmse']*100:.1f}%|")
    gains=['|翼型|平均 Fz 系数|周期 Fx 系数|周期 Fy 系数|周期 Fz 系数|','|---|---:|---:|---:|---:|']
    for kind in ('curve','rect'):
        g=wg[kind]['axis_gains']
        gains.append(f"|{kind}|{fits[kind]['gain']:.3f}|{g['Fx']:.3f}|{g['Fy']:.3f}|{g['Fz']:.3f}|")
    moment=['|翼型|分量|单独均值拟合的非负系数（条件诊断）|','|---|---|---:|']
    for row in mean6:
        moment.append(f"|{row['kind']}|{row['axis']}|{row['diagnostic_axis_gain']:.3g}|")
    text=f'''# 系数能否解决实测与仿真误差？

## Material Passport
模式：已有实验结果的描述性校准检查。Verification Status: ANALYZED。
使用两翼型各 18 次实测及此前 ±26.1°、单翼、固定基座仿真。原始记录及物理模型均未修改。
本次是输出系数拟合，不是修改气动系数后的动力学重跑。

## 结论
**每种翼型各用一个系数，可以改善平均 Fz 在本次 10–15 Hz 范围内的预测；不能据此认定 feathering angle 是误差来源，也不能证明六维力准确。**
两种翼型需要不同系数；周期波形与均值需要的系数也不同。校准不能自动转移到其他翼型、运动状态或力矩。

## 1. 平均升力：用留出频率检验
模型为 Fz_corrected=k Fz_sim，k≥0，无截距。仅使用 10、11、12 Hz 的三次重复均值拟合，13、14、15 Hz 固定 k 后评价。
两个翼型共用系数时按力的平方误差最小化，较大升力的矩形翼权重自然较大；不是按各点百分比加权。
下表不代表总体误差保证，也未预设“准确”阈值。

{chr(10).join(lines)}

![平均升力校准](lift_gain_holdout.png)

三次重复误差棒是 SD，不是校准置信区间。基线继续采用运动段 0.5–4 s 减尾段 5.5 s 至末尾前 0.1 s；停机去皮假设仍待确认。
`lift_holdout.csv` 含所有点及校正前后数值；`summary.json` 还列出跨翼型迁移误差。

## 2. 平均值校准不等于周期载荷校准
坐标 yaw 固定为此前只用 10–12 Hz 拟合的候选值：曲面 −27.5°，矩形 −31.5°。Z 保持不变。
每次记录沿用同一个此前确定的时间相位；不为不同校正方式重新优化相位或 yaw。
比较 1–4 阶周期波形，使用总线动量重建仿真支座惯性反力。留出记录仍允许此前的时间对齐，并非绝对时钟预测。

{chr(10).join(gains)}

{chr(10).join(wave)}

单轴 NRMSE=RMSE/实测该轴交流 RMS；向量 NRMSE 按三轴合并 RMS 归一化，均先逐次记录计算再平均。
非零均值误差仍包含在 RMSE 中。上述分母使小幅轴更敏感，因此同时在 CSV 保存 mN 的绝对 RMSE。
逐轴系数是较宽松的诊断，不是统一物理修正；它仍不能改变谐波比例、峰形、相位关系。

![周期波形](force_gain_waveforms.png)

“整个力乘系数”会同时放大机构惯性载荷；这不是单纯调整气动系数。
“只缩放气动力”一项使用 F_fixture,new=F_fixture,old+(k−1)F_aero,old，保留既有运动，
是固定运动条件下的诊断。真实修改流体参数会改变被动羽化轨迹，需要再求解动力学，不能把此项称为已验证新物理模型。

## 3. 为什么不能由系数推出 feathering angle
平均力是整个周期气动力的积分。几何、速度、气动系数、羽化时序、形变以及测量基线的差异，都可能表现为平均值的比例误差。
目前只知道羽化限位 ±26.1°；它不能给出真实 α(t)、换向时间或角速度。乘上 k 没有识别这些运动变量。
即使平均值完全重合，峰值、谐波和六维载荷仍可能不同。因此可暂用经验系数估计此范围内的平均升力，不能以此忽略羽化运动验证。

## 4. 六维力矩的条件检查
六维量为 [Fx,Fy,Fz,Mx,My,Mz]。力矩需要统一测量原点：转换包含旋转及 r×F 项，不能只旋转坐标或乘系数。
传感器相对 base_link 的位置、静态去皮姿态尚未提供。此前 2 kHz 直接支座力矩还可能受限位冲击采样影响，未记录可用于稳定重建的总角动量。
因此本次 **不将三个力矩分量判为已验证**。

为检查“一个比例系数能否解释全部均值”，在假设原点重合、重力基线已消除的条件下，也计算了每轴各自的系数：

{chr(10).join(moment)}

这里只把仿真气动平均力矩作为条件预测；实际支座力矩还可能含运动导致的平均重力矩变化。
系数为零表示非负最小二乘落在边界（例如符号相反），不能解释成真实材料/传感器增益为零。
这一表用于暴露统一系数假设的限制，不能据此给力矩传感器重新标定。

## 可支持的表述
“针对当前单翼固定基座工况，在 10–12 Hz 标定每种翼型的平均升力修正系数后，13–15 Hz 的平均 Fz 误差为上表所列范围。”
目前不支持“误差已经确定来自羽化角，可忽略其影响，仿真能够准确获得六维力”。

## 检查与复现
11/11 统计解释检查：分翼型报告以避免混合趋势；重复实验而非采样点作为单位；全纳入无选择/幸存者筛除；
无碰撞变量控制、诊断基率或极值筛选前后比较；所有频率/轴/拟合方式公开；探索性校准不作因果或反向因果推断；
拟合/留出频率分开，没有使用留出力数据更新系数。共计零项显著性检验。
全部结论继承此前的质量、翼膜边界、安装偏移、频率不同步和尾段基线假设。

```bash
python experiments/fixbase_curved_real/coefficient_validation.py
```

`summary.json` 包含输入与分析脚本哈希。此分析不覆盖既有的原始实验或未校正仿真结果。
'''
    (OUTPUT/'report.md').write_text(text,encoding='utf-8')


def notebook():
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    n=nbf.v4.new_notebook()
    n.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    n.cells=[nbf.v4.new_markdown_cell('# 系数校准与留出验证\n\n输出校准实验；10–12 Hz 拟合，13–15 Hz 验证。并非修改气动参数后的重跑。'),
             nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import display, Image, Markdown
here=Path.cwd().resolve()
ROOT=next(p for p in (here,*here.parents) if (p/'flapsim/simulator.py').exists())
sys.path.insert(0,str(ROOT/'experiments/fixbase_curved_real'))
from coefficient_validation import main, OUTPUT
result=main()
report=(OUTPUT/'report.md').read_text()
display(Markdown(report.replace('![平均升力校准](lift_gain_holdout.png)','').replace('![周期波形](force_gain_waveforms.png)','')))'''),
             nbf.v4.new_code_cell("display(Image(filename=str(OUTPUT/'lift_gain_holdout.png')))\ndisplay(Image(filename=str(OUTPUT/'force_gain_waveforms.png')))")]
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(n,timeout=120,km=km,resources={'metadata':{'path':str(FOLDER)}}).execute()
    nbf.write(n,FOLDER/'coefficient_validation.ipynb')


if __name__=='__main__':
    main()
    notebook()
