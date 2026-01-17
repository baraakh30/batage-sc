#!/usr/bin/env python3
"""
Debug script for BATAGE-SC predictor.

Analyzes what's helping and what's hurting vs baseline TAGE.
Shows:
1. SC override effectiveness (helps vs hurts)
2. Branch type classification stats
3. Per-branch analysis of mispredictions
4. Comparison with baseline TAGE on specific patterns
"""

import sys
import os
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictors.tage_original import OriginalTAGE, TAGE_8C_64KB
from src.predictors.batage_sc import BATAGE_SC, BATAGE_SC_64KB, BranchType
from src.trace.parser import TraceParser


class DebugBATAGE(BATAGE_SC):
    """Extended BATAGE-SC with detailed debugging statistics."""
    
    def __init__(self, config=None):
        super().__init__(config or BATAGE_SC_64KB)
        
        # SC override tracking
        self.sc_override_helped = 0
        self.sc_override_hurt = 0
        self.sc_override_neutral = 0
        
        # TAGE confidence tracking
        self.tage_low_conf_correct = 0
        self.tage_low_conf_wrong = 0
        self.tage_high_conf_correct = 0
        self.tage_high_conf_wrong = 0
        
        # Branch type tracking
        self.branch_type_correct = {BranchType.UNKNOWN: 0, BranchType.LOOP_LIKE: 0, BranchType.DATA_DEPENDENT: 0}
        self.branch_type_wrong = {BranchType.UNKNOWN: 0, BranchType.LOOP_LIKE: 0, BranchType.DATA_DEPENDENT: 0}
        
        # Per-PC tracking (for identifying problematic branches)
        self.pc_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'sc_helped': 0, 'sc_hurt': 0})
        
        # Provider table tracking
        self.provider_correct = defaultdict(int)
        self.provider_wrong = defaultdict(int)
        
        # Track last N problematic predictions
        self.problem_history = []
        self.max_problem_history = 100
    
    def update(self, pc: int, history: np.ndarray, taken: bool, prediction):
        """Override update to track detailed statistics."""
        info = self._pred_info
        
        tage_pred = info['tage_pred']
        final_pred = info['final_pred']
        use_sc = info['use_sc']
        provider = info['provider']
        provider_conf = info['provider_conf']
        branch_type = info['branch_type']
        sc_pred = info['sc_pred']
        
        # Track SC override effectiveness
        if use_sc:
            tage_would_be_correct = (tage_pred == taken)
            sc_is_correct = (final_pred == taken)
            
            if sc_is_correct and not tage_would_be_correct:
                self.sc_override_helped += 1
                self.pc_stats[pc]['sc_helped'] += 1
            elif not sc_is_correct and tage_would_be_correct:
                self.sc_override_hurt += 1
                self.pc_stats[pc]['sc_hurt'] += 1
                # Record problematic SC override
                if len(self.problem_history) < self.max_problem_history:
                    self.problem_history.append({
                        'type': 'sc_hurt',
                        'pc': pc,
                        'tage_pred': tage_pred,
                        'sc_pred': sc_pred,
                        'actual': taken,
                        'provider': provider,
                        'provider_conf': provider_conf,
                        'branch_type': branch_type,
                    })
            else:
                self.sc_override_neutral += 1
        
        # Track TAGE confidence
        if provider_conf < self.sc_confidence_threshold:
            if tage_pred == taken:
                self.tage_low_conf_correct += 1
            else:
                self.tage_low_conf_wrong += 1
        else:
            if tage_pred == taken:
                self.tage_high_conf_correct += 1
            else:
                self.tage_high_conf_wrong += 1
        
        # Track branch type effectiveness
        if final_pred == taken:
            self.branch_type_correct[branch_type] += 1
            self.pc_stats[pc]['correct'] += 1
        else:
            self.branch_type_wrong[branch_type] += 1
            self.pc_stats[pc]['wrong'] += 1
        
        # Track provider table
        if provider >= 0:
            if final_pred == taken:
                self.provider_correct[provider] += 1
            else:
                self.provider_wrong[provider] += 1
        else:
            if final_pred == taken:
                self.provider_correct[-1] += 1  # -1 = bimodal
            else:
                self.provider_wrong[-1] += 1
        
        # Call parent update
        super().update(pc, history, taken, prediction)
    
    def get_debug_stats(self):
        """Get comprehensive debug statistics."""
        total_sc = self.sc_override_helped + self.sc_override_hurt + self.sc_override_neutral
        
        stats = {
            'sc_overrides': {
                'total': total_sc,
                'helped': self.sc_override_helped,
                'hurt': self.sc_override_hurt,
                'neutral': self.sc_override_neutral,
                'help_rate': self.sc_override_helped / total_sc * 100 if total_sc > 0 else 0,
                'hurt_rate': self.sc_override_hurt / total_sc * 100 if total_sc > 0 else 0,
                'net_benefit': self.sc_override_helped - self.sc_override_hurt,
            },
            'tage_confidence': {
                'low_conf_correct': self.tage_low_conf_correct,
                'low_conf_wrong': self.tage_low_conf_wrong,
                'low_conf_accuracy': self.tage_low_conf_correct / (self.tage_low_conf_correct + self.tage_low_conf_wrong) * 100 if (self.tage_low_conf_correct + self.tage_low_conf_wrong) > 0 else 0,
                'high_conf_correct': self.tage_high_conf_correct,
                'high_conf_wrong': self.tage_high_conf_wrong,
                'high_conf_accuracy': self.tage_high_conf_correct / (self.tage_high_conf_correct + self.tage_high_conf_wrong) * 100 if (self.tage_high_conf_correct + self.tage_high_conf_wrong) > 0 else 0,
            },
            'branch_types': {},
            'provider_tables': {},
            'worst_branches': [],
            'problem_history': self.problem_history,
        }
        
        # Branch type stats
        for bt in [BranchType.UNKNOWN, BranchType.LOOP_LIKE, BranchType.DATA_DEPENDENT]:
            c = self.branch_type_correct[bt]
            w = self.branch_type_wrong[bt]
            name = ['UNKNOWN', 'LOOP_LIKE', 'DATA_DEPENDENT'][bt]
            stats['branch_types'][name] = {
                'correct': c,
                'wrong': w,
                'total': c + w,
                'accuracy': c / (c + w) * 100 if (c + w) > 0 else 0
            }
        
        # Provider table stats
        for p in sorted(self.provider_correct.keys()):
            c = self.provider_correct[p]
            w = self.provider_wrong[p]
            name = f'Table_{p}' if p >= 0 else 'Bimodal'
            stats['provider_tables'][name] = {
                'correct': c,
                'wrong': w,
                'total': c + w,
                'accuracy': c / (c + w) * 100 if (c + w) > 0 else 0
            }
        
        # Find worst branches (most mispredictions)
        worst = sorted(self.pc_stats.items(), key=lambda x: x[1]['wrong'], reverse=True)[:20]
        for pc, s in worst:
            total = s['correct'] + s['wrong']
            stats['worst_branches'].append({
                'pc': hex(pc),
                'mispredictions': s['wrong'],
                'total': total,
                'accuracy': s['correct'] / total * 100 if total > 0 else 0,
                'sc_helped': s['sc_helped'],
                'sc_hurt': s['sc_hurt'],
            })
        
        return stats


def run_debug_comparison(trace_path: Path, max_branches: int = 500000, verbose: bool = True):
    """Run BATAGE-SC and TAGE side by side with debugging."""
    
    print(f"\n{'='*70}")
    print(f"Debug Analysis: {trace_path.name}")
    print(f"{'='*70}")
    
    # Create predictors
    debug_batage = DebugBATAGE(BATAGE_SC_64KB)
    tage = OriginalTAGE(TAGE_8C_64KB)
    
    # Warmup
    warmup = 50000
    
    # Stats
    batage_correct = 0
    batage_wrong = 0
    tage_correct = 0
    tage_wrong = 0
    
    # Disagreement tracking
    disagreements = {'batage_right_tage_wrong': 0, 'tage_right_batage_wrong': 0, 'both_wrong': 0, 'both_right': 0}
    
    # Use proper trace parser
    parser = TraceParser(format_name='cbp2025')
    
    branch_count = 0
    history = np.zeros(4096, dtype=np.int8)
    
    for record in parser.parse_file(trace_path, max_branches=max_branches + warmup):
        pc = record.pc
        taken = record.taken
        
        branch_count += 1
        
        # Get predictions
        batage_pred = debug_batage.predict(pc, history)
        tage_pred = tage.predict(pc, history)
        
        if branch_count > warmup:
            # Track accuracy
            if batage_pred.prediction == taken:
                batage_correct += 1
            else:
                batage_wrong += 1
            
            if tage_pred.prediction == taken:
                tage_correct += 1
            else:
                tage_wrong += 1
            
            # Track disagreements
            b_right = batage_pred.prediction == taken
            t_right = tage_pred.prediction == taken
            
            if b_right and t_right:
                disagreements['both_right'] += 1
            elif b_right and not t_right:
                disagreements['batage_right_tage_wrong'] += 1
            elif not b_right and t_right:
                disagreements['tage_right_batage_wrong'] += 1
            else:
                disagreements['both_wrong'] += 1
        
        # Update predictors
        debug_batage.update(pc, history, taken, batage_pred)
        tage.update(pc, history, taken, tage_pred)
        
        # Update history
        history = np.roll(history, 1)
        history[0] = 1 if taken else 0
        
        if verbose and branch_count % 100000 == 0:
            print(f"  Processed {branch_count:,} branches...")
    
    # Calculate accuracies
    batage_acc = batage_correct / (batage_correct + batage_wrong) * 100 if (batage_correct + batage_wrong) > 0 else 0
    tage_acc = tage_correct / (tage_correct + tage_wrong) * 100 if (tage_correct + tage_wrong) > 0 else 0
    
    print(f"\n--- Overall Results ---")
    print(f"BATAGE-SC: {batage_acc:.4f}% ({batage_wrong:,} mispredictions)")
    print(f"TAGE:      {tage_acc:.4f}% ({tage_wrong:,} mispredictions)")
    print(f"Difference: {batage_acc - tage_acc:+.4f}%")
    
    print(f"\n--- Disagreement Analysis ---")
    total = sum(disagreements.values())
    print(f"Both correct:         {disagreements['both_right']:>8,} ({disagreements['both_right']/total*100:.2f}%)")
    print(f"Both wrong:           {disagreements['both_wrong']:>8,} ({disagreements['both_wrong']/total*100:.2f}%)")
    print(f"BATAGE right, TAGE wrong: {disagreements['batage_right_tage_wrong']:>5,} ({disagreements['batage_right_tage_wrong']/total*100:.2f}%) <- BATAGE wins")
    print(f"TAGE right, BATAGE wrong: {disagreements['tage_right_batage_wrong']:>5,} ({disagreements['tage_right_batage_wrong']/total*100:.2f}%) <- BATAGE loses")
    print(f"Net wins for BATAGE:  {disagreements['batage_right_tage_wrong'] - disagreements['tage_right_batage_wrong']:+,}")
    
    # Get debug stats
    stats = debug_batage.get_debug_stats()
    
    print(f"\n--- SC Override Analysis ---")
    sc = stats['sc_overrides']
    print(f"Total SC overrides:   {sc['total']:,}")
    print(f"  Helped (saved):     {sc['helped']:,} ({sc['help_rate']:.2f}%)")
    print(f"  Hurt (caused miss): {sc['hurt']:,} ({sc['hurt_rate']:.2f}%)")
    print(f"  Neutral:            {sc['neutral']:,}")
    print(f"  Net benefit:        {sc['net_benefit']:+,}")
    if sc['total'] > 0:
        print(f"  ** SC is {'HELPING' if sc['net_benefit'] > 0 else 'HURTING'} overall **")
    
    print(f"\n--- TAGE Confidence Analysis ---")
    tc = stats['tage_confidence']
    print(f"Low confidence predictions (conf < {debug_batage.sc_confidence_threshold}):")
    print(f"  Accuracy: {tc['low_conf_accuracy']:.2f}% ({tc['low_conf_correct']:,} / {tc['low_conf_correct'] + tc['low_conf_wrong']:,})")
    print(f"High confidence predictions:")
    print(f"  Accuracy: {tc['high_conf_accuracy']:.2f}% ({tc['high_conf_correct']:,} / {tc['high_conf_correct'] + tc['high_conf_wrong']:,})")
    
    print(f"\n--- Branch Type Classification ---")
    for name, bt_stats in stats['branch_types'].items():
        if bt_stats['total'] > 0:
            print(f"{name:15}: {bt_stats['accuracy']:.2f}% ({bt_stats['correct']:,} / {bt_stats['total']:,})")
    
    print(f"\n--- Provider Table Performance ---")
    for name, pt_stats in stats['provider_tables'].items():
        if pt_stats['total'] > 1000:  # Only show significant tables
            print(f"{name:10}: {pt_stats['accuracy']:.2f}% ({pt_stats['correct']:,} / {pt_stats['total']:,})")
    
    print(f"\n--- Top 10 Worst Branches ---")
    print(f"{'PC':<18} {'Misses':>8} {'Total':>10} {'Acc':>8} {'SC Help':>8} {'SC Hurt':>8}")
    print("-" * 70)
    for b in stats['worst_branches'][:10]:
        print(f"{b['pc']:<18} {b['mispredictions']:>8,} {b['total']:>10,} {b['accuracy']:>7.2f}% {b['sc_helped']:>8,} {b['sc_hurt']:>8,}")
    
    if stats['problem_history']:
        print(f"\n--- Sample SC Override Failures ---")
        hurt_cases = [p for p in stats['problem_history'] if p['type'] == 'sc_hurt'][:5]
        for p in hurt_cases:
            bt_name = ['UNKNOWN', 'LOOP_LIKE', 'DATA_DEP'][p['branch_type']]
            print(f"  PC {hex(p['pc'])}: TAGE={p['tage_pred']} SC={p['sc_pred']} Actual={p['actual']} "
                  f"Provider={p['provider']} Conf={p['provider_conf']:.2f} Type={bt_name}")
    
    return {
        'trace': trace_path.name,
        'batage_acc': batage_acc,
        'tage_acc': tage_acc,
        'diff': batage_acc - tage_acc,
        'sc_stats': stats['sc_overrides'],
        'disagreements': disagreements,
    }


def main():
    parser = argparse.ArgumentParser(description='Debug BATAGE-SC predictor')
    parser.add_argument('--trace', type=str, help='Specific trace file to analyze')
    parser.add_argument('--category', type=str, help='Category to analyze (int, fp, web, etc.)')
    parser.add_argument('--max-branches', type=int, default=500000, help='Max branches per trace')
    parser.add_argument('--all', action='store_true', help='Run on all traces')
    
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'branch'
    
    if args.trace:
        trace_path = Path(args.trace)
        if not trace_path.exists():
            # Try finding in data dir
            for cat_dir in data_dir.iterdir():
                if cat_dir.is_dir():
                    for f in cat_dir.glob(f'*{args.trace}*'):
                        trace_path = f
                        break
        
        if trace_path.exists():
            run_debug_comparison(trace_path, args.max_branches)
        else:
            print(f"Trace not found: {args.trace}")
    
    elif args.category:
        cat_dir = data_dir / args.category
        if cat_dir.exists():
            results = []
            for trace in sorted(cat_dir.glob('*_trace.gz'))[:4]:
                r = run_debug_comparison(trace, args.max_branches)
                results.append(r)
            
            # Summary
            print(f"\n{'='*70}")
            print(f"CATEGORY SUMMARY: {args.category}")
            print(f"{'='*70}")
            for r in results:
                sc = r['sc_stats']
                marker = '✓' if r['diff'] > 0 else '✗' if r['diff'] < 0 else '='
                print(f"{marker} {r['trace']:25} BATAGE: {r['batage_acc']:.2f}%  TAGE: {r['tage_acc']:.2f}%  "
                      f"Diff: {r['diff']:+.2f}%  SC net: {sc['net_benefit']:+}")
        else:
            print(f"Category not found: {args.category}")
    
    elif args.all:
        all_results = []
        for cat_dir in sorted(data_dir.iterdir()):
            if cat_dir.is_dir():
                for trace in sorted(cat_dir.glob('*_trace.gz'))[:2]:  # 2 per category
                    r = run_debug_comparison(trace, args.max_branches, verbose=False)
                    all_results.append(r)
        
        # Overall summary
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")
        
        total_helped = sum(r['sc_stats']['helped'] for r in all_results)
        total_hurt = sum(r['sc_stats']['hurt'] for r in all_results)
        
        print(f"\nSC Override Total: Helped={total_helped:,}  Hurt={total_hurt:,}  Net={total_helped-total_hurt:+,}")
        
        wins = sum(1 for r in all_results if r['diff'] > 0.01)
        losses = sum(1 for r in all_results if r['diff'] < -0.01)
        ties = len(all_results) - wins - losses
        
        print(f"Trace Results: Wins={wins}  Losses={losses}  Ties={ties}")
        
    else:
        # Default: analyze web traces (where we're losing)
        print("Analyzing WEB category (where BATAGE-SC is losing)...")
        cat_dir = data_dir / 'web'
        if cat_dir.exists():
            for trace in sorted(cat_dir.glob('*_trace.gz'))[:2]:
                run_debug_comparison(trace, args.max_branches)


if __name__ == '__main__':
    main()
