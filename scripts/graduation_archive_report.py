"""Run only read-only archive surfaces; keep structured artifacts and a GitHub summary."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain',choices=['both','solana','robinhood'],default='both')
    parser.add_argument('--mode',choices=['all','daily','backlog','report'],default='all')
    parser.add_argument('--hours',type=int,default=48)
    parser.add_argument('--cohort-since',default='',help='optional ISO graduation cutoff with timezone')
    parser.add_argument('--output',type=Path,default=Path('archive-reports'))
    args=parser.parse_args()
    if not 1<=args.hours<=720:parser.error('--hours must be between 1 and 720')
    if args.cohort_since and args.mode not in ('all','report'):
        parser.error('--cohort-since requires all or report mode')
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    sections=['# Graduation Archive report',
              'Read-only. Historical gaps and unknown chain attribution are retained.']
    for chain in (['robinhood','solana'] if args.chain=='both' else [args.chain]):
        for mode in (['daily','backlog','report'] if args.mode=='all' else [args.mode]):
            command=[sys.executable,str(root/'launchpad_archive.py'),'--'+mode,'--chain',chain,
                     '--hours',str(args.hours),'--days',str(math.ceil(args.hours/24))]
            if mode=='report' and args.cohort_since:command+=['--cohort-since',args.cohort_since]
            # Keep child diagnostics visible; a failed connection must not become
            # an opaque CalledProcessError with no actionable cause.
            result=subprocess.run(command,cwd=root,check=True,stdout=subprocess.PIPE,text=True)
            payload=json.loads(result.stdout)
            rendered=json.dumps(payload,indent=2)
            (args.output/f'{chain}-{mode}.json').write_text(rendered+'\n',encoding='utf-8')
            sections.append(f'## {chain} — {mode}\n\n```json\n{rendered}\n```')
    summary='\n\n'.join(sections)+'\n'
    (args.output/'summary.md').write_text(summary,encoding='utf-8')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a',encoding='utf-8') as target:target.write(summary)
    print(summary)


if __name__=='__main__':main()
