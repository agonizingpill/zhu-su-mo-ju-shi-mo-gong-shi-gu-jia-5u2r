import random
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, func
from src.db import Base,engine,Session
from src.models import User,Mould,MachineSlot,Estimate,TrialHistory
from src.execution_models import TrialExecution,CostReview
from src.workshop_models import MaintenanceWindow,ResinLot,TrialMaterial,ReviewMaterialLedger,TrialDelivery
from src.booking_models import YieldModel,SampleOrder,SampleDispatch
from src.booking_service import train_models,create_order
from src.auth import hash_password

UTC=timezone.utc

def _signed_trial(s,code,estimate,mould,slot,started,good,total,rng):
    if s.scalar(select(Estimate).where(Estimate.code==code)):
        return
    hours=round(6+rng.random()*5,2)
    rate=300.0
    e=Estimate(code=code,mould_id=mould.id,slot_id=slot.id,rate=rate,setup_hours=1,
               predicted_hours=hours,quote_amount=round(hours*rate*1.2,2),
               status='committed',model_result={'note':'历史已签试模，用于按模具训练良品率下界'})
    s.add(e);s.flush()
    x=TrialExecution(estimate_id=e.id,started_at=started.isoformat(),
                     ended_at=(started+timedelta(hours=hours)).isoformat(),
                     actual_hours=hours,good_shots=good,total_shots=total,
                     snapshot={'estimate_code':e.code,'predicted_hours':hours,
                               'quoted_amount':e.quote_amount,'rate':rate,
                               'mould_code':mould.code,'slot_id':slot.id},
                     status='settled')
    s.add(x);s.flush()
    material=round(300+rng.random()*200,2)
    actual=round(hours*rate+material,2)
    review=CostReview(execution_id=x.id,material_cost=material,actual_cost=actual,
                      quoted_amount=e.quote_amount,
                      variance_hours=0.0,variance_pct=0.0,
                      margin=round(e.quote_amount-actual,2),
                      reason='历史签结实绩，作为良品率训练样本',status='signed',
                      signed_by='admin@ruimo.demo')
    s.add(review);s.flush()
    s.add(ReviewMaterialLedger(review_id=review.id,net_cost=material,
                               lines=[{'note':'历史净耗快照','net_cost':material}]))

def seed():
    Base.metadata.create_all(engine)
    with Session() as s:
        for role in ('admin','estimator','viewer'):
            if not s.scalar(select(User).where(User.email==role+'@ruimo.demo')): s.add(User(email=role+'@ruimo.demo',role=role,password_hash=hash_password('Mould2026!')))
        for code,name,cav,slides,steel,kg in [('MD-86','八腔医疗接头模',8,4,'S136',860),('MD-42','双腔仪表外壳模',2,2,'P20',420),('MD-115','十六腔盖件模',16,6,'H13',1150)]:
            if not s.scalar(select(Mould).where(Mould.code==code)): s.add(Mould(code=code,name=name,cavities=cav,slides=slides,steel=steel,weight_kg=kg))
        s.flush()
        for machine,start,cap in [('M280','2026-11-18 08:00',48),('M280','2026-11-19 08:00',4),('M450','2026-11-18 08:00',72)]:
            if not s.scalar(select(MachineSlot).where(MachineSlot.machine==machine,MachineSlot.start_hour==start)): s.add(MachineSlot(machine=machine,start_hour=start,capacity_hours=cap,reserved_hours=0))
        rng=random.Random(42)
        for i in range(160):
            code=f'TR-{i+1:03d}'
            if s.scalar(select(TrialHistory).where(TrialHistory.trial_code==code)):continue
            cav=rng.choice([2,4,8,16]);sl=rng.randint(0,8);kg=rng.uniform(180,1600);steel=rng.choice(['P20','H13','S136'])
            hours=2+cav*.3+sl*.65+kg*.004+cav*sl*.035+(['P20','H13','S136'].index(steel))*.8+rng.gauss(0,.4)
            s.add(TrialHistory(trial_code=code,cavities=cav,slides=sl,steel=steel,weight_kg=round(kg,1),actual_hours=round(hours,2)))
        s.flush()
        if not s.scalar(select(Estimate).where(Estimate.code=='QE-26017')):
            m=s.scalar(select(Mould).where(Mould.code=='MD-86'))
            s.add(Estimate(code='QE-26017',mould_id=m.id,rate=280,setup_hours=2,quote_amount=8000,status='draft'))
        s.flush()
        if not s.scalar(select(Estimate).where(Estimate.code=='QE-FLOOR-DEMO')):
            m=s.scalar(select(Mould).where(Mould.code=='MD-42'));slot=s.scalar(select(MachineSlot).where(MachineSlot.machine=='M450'))
            s.add(Estimate(code='QE-FLOOR-DEMO',mould_id=m.id,slot_id=slot.id,rate=300,setup_hours=1,predicted_hours=10,quote_amount=5000,status='committed',model_result={'note':'已承诺的演示订单，供实绩采集；不作模型效果指标'}));slot.reserved_hours+=10
        for code,grade,qty,price,state in [('RS-P20-DEMO','ABS自然色',500,20,'available'),('RS-ISOLATED','ABS复检隔离',50,18,'quarantine')]:
            if not s.scalar(select(ResinLot).where(ResinLot.code==code)):
                s.add(ResinLot(code=code,grade=grade,received_kg=qty,available_kg=qty,unit_price=price,state=state))

        # ---- V4: per-mould signed actuals for good-rate lower-bound training ----
        rng2=random.Random(20261120)
        mould_codes=('MD-86','MD-42','MD-115')
        # Observed good rates differ by mould so each group gets its own bound.
        rate_profiles={'MD-86':(0.94,0.02),'MD-42':(0.88,0.035),'MD-115':(0.82,0.05)}
        slot_m280=s.scalar(select(MachineSlot).where(MachineSlot.machine=='M280',MachineSlot.start_hour=='2026-11-18 08:00'))
        base=datetime(2026,9,1,8,0,tzinfo=UTC)
        for gi,mcode in enumerate(mould_codes):
            mould=s.scalar(select(Mould).where(Mould.code==mcode))
            mu,sigma=rate_profiles[mcode]
            for k in range(6):
                code=f'QE-HIST-{mcode[-3:]}-{k+1:02d}'
                if s.scalar(select(Estimate).where(Estimate.code==code)):
                    continue
                total=120
                rate=min(0.99,max(0.5,rng2.gauss(mu,sigma)))
                good=int(round(total*rate))
                started=base+timedelta(days=gi*10+k*2)
                _signed_trial(s,code,None,mould,slot_m280,started,good,total,rng2)
        s.flush()

        class _Actor:
            email='seed@ruimo.demo';role='admin'
        actor=_Actor()
        if not s.scalar(select(YieldModel)):
            for mould in s.scalars(select(Mould).order_by(Mould.id)):
                from src.booking_schemas import TrainInput
                train_models(s,TrainInput(mould_id=mould.id),actor)

        # ---- V4: demo booking orders against the not-yet-finished floor trial ----
        floor=s.scalar(select(Estimate).where(Estimate.code=='QE-FLOOR-DEMO'))
        if floor and not s.scalar(select(SampleOrder).where(SampleOrder.order_no=='SO-261101')):
            from src.booking_schemas import OrderInput
            create_order(s,OrderInput(order_no='SO-261101',customer='华远质量部',
                                      estimate_id=floor.id,planned_shots=100,
                                      requested_qty=50,note='首批订样，完工前承诺'),actor)
            create_order(s,OrderInput(order_no='SO-261102',customer='华远装配线',
                                      estimate_id=floor.id,planned_shots=100,
                                      requested_qty=30,note='并发订样，同池不超占'),actor)
        # A fourth mould without signed history keeps its order suspended.
        if not s.scalar(select(Mould).where(Mould.code=='MD-77')):
            s.add(Mould(code='MD-77',name='四腔演示挂起模',cavities=4,slides=1,steel='P20',weight_kg=300));s.flush()
        m77=s.scalar(select(Mould).where(Mould.code=='MD-77'))
        if not s.scalar(select(Estimate).where(Estimate.code=='QE-PENDING-DEMO')):
            s.add(Estimate(code='QE-PENDING-DEMO',mould_id=m77.id,slot_id=slot_m280.id,rate=260,
                           setup_hours=1,predicted_hours=8,quote_amount=4000,status='committed',
                           model_result={'note':'样本不足演示估单'}));s.flush()
        pe=s.scalar(select(Estimate).where(Estimate.code=='QE-PENDING-DEMO'))
        if pe and not s.scalar(select(SampleOrder).where(SampleOrder.order_no=='SO-261199')):
            from src.booking_schemas import OrderInput
            create_order(s,OrderInput(order_no='SO-261199',customer='样本不足客户',
                                      estimate_id=pe.id,planned_shots=80,requested_qty=40,
                                      note='该模具已签实绩不足，应挂起'),actor)
        s.commit()
        print({t.__tablename__:s.scalar(select(func.count()).select_from(t)) for t in (User,Mould,MachineSlot,Estimate,TrialHistory,TrialExecution,CostReview,MaintenanceWindow,ResinLot,TrialMaterial,ReviewMaterialLedger,TrialDelivery,YieldModel,SampleOrder,SampleDispatch)})
if __name__=='__main__':seed()
