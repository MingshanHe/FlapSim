"""Experimental mass-distribution variants; never overwrite the shared MJCF."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from flapsim import Simulator,load_config
from flapsim.rl.env import PRESET
from .hover_initial import HoverInitialEnv


def make_variant(path,upper_g=7.,lower_g=9.):
    source=load_config(PRESET).path
    tree=ET.parse(source); root=tree.getroot()
    compiler=root.find('compiler')
    compiler.set('meshdir',str((source.parent/compiler.get('meshdir','.')).resolve()))
    thrust=root.find(".//body[@name='thrust_unit']")
    mount=root.find(".//body[@name='base_link']/inertial")
    body=thrust.find('inertial')
    moving=sum(float(i.get('mass')) for i in thrust.findall('.//body/inertial'))
    connector=float(root.find(".//body[@name='pitch_link']/inertial").get('mass'))
    changes=[]
    for name,inertial,new_mass in (('thrust_unit',body,upper_g/1000.-moving),('base_link',mount,lower_g/1000.-connector)):
        old_mass=float(inertial.get('mass'))
        if new_mass<=0: raise ValueError('specified group mass leaves no positive fixed-structure mass')
        scale=new_mass/old_mass
        inertial.set('mass',f'{new_mass:.17g}')
        attribute='fullinertia' if 'fullinertia' in inertial.attrib else 'diaginertia'
        inertial.set(attribute,' '.join(f'{float(v)*scale:.17g}' for v in inertial.get(attribute).split()))
        changes.append(dict(body=name,old_mass_g=1000*old_mass,new_mass_g=1000*new_mass,inertia_scale=scale))
    root.set('model',f'IST-V2-upper{upper_g:g}-lower{lower_g:g}')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tree.write(path,encoding='utf-8',xml_declaration=True)
    return dict(path=str(path.resolve()),upper_g=upper_g,lower_g=lower_g,total_g=upper_g+lower_g,changes=changes,
                assumption='Keep all body COM locations and moving components unchanged; scale only fixed-structure inertia in proportion to mass; lower includes 1 g pitch connector.')


def model_config(path): return replace(load_config(PRESET),model_path=Path(path))


class MassVariantEnv(HoverInitialEnv):
    def __init__(self,trim,model_path,**kwargs):
        super().__init__(trim,simulation_config=model_config(model_path),**kwargs)


def describe(path):
    config=model_config(path)
    sim=Simulator(replace(config,scene=replace(config.scene,orientation=(1.,0.,0.,0.))))
    model,data=sim.model,sim.data
    base=model.body('base_link').id; upper=model.body('thrust_unit').id
    def descendants(ancestor):
        selected=[]
        for i in range(1,model.nbody):
            j=i
            while j and j!=ancestor: j=int(model.body_parentid[j])
            if j==ancestor: selected.append(i)
        return selected
    top=descendants(upper); whole=descendants(base); lower=[i for i in whole if i not in top]
    origin=data.xpos[base]
    result={}
    for name,ids in (('upper',top),('lower',lower),('total',whole)):
        masses=model.body_mass[ids]; mass=masses.sum()
        com=np.sum(masses[:,None]*data.xipos[ids],axis=0)/mass
        inertia=np.zeros((3,3))
        for i in ids:
            rotation=data.ximat[i].reshape(3,3); arm=data.xipos[i]-com
            inertia+=rotation@np.diag(model.body_inertia[i])@rotation.T+model.body_mass[i]*(np.dot(arm,arm)*np.eye(3)-np.outer(arm,arm))
        result[name]=dict(mass_g=float(mass*1000),com_base_mm=((com-origin)*1000).tolist(),inertia_com_kg_m2=inertia.tolist(),
                          bodies=[dict(name=model.body(i).name,mass_g=float(model.body_mass[i]*1000),com_base_mm=((data.xipos[i]-origin)*1000).tolist()) for i in ids])
    result['pivots_base_mm']={name:((data.xanchor[model.joint(name).id]-origin)*1000).tolist() for name in ('pitch_joint','roll_joint')}
    return result
