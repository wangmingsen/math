"""Read-only Q2 contract check before interpreting model experiments.

Text padding is given by the BERT attention mask. For audio/vision, zero rows
are observed numeric values with unknown cause; their true padding and missing
masks are not supplied. Only injected masks have known missing positions.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from aligned_data import load_small, normalize


def lexical_check(tokens,raw_text,tokenizer):
    ids=np.asarray(tokens[0]);attention=np.asarray(tokens[1]);types=np.asarray(tokens[2])
    valid=attention!=0
    encoded=tokenizer.encode(str(raw_text))
    observed=ids[valid].astype(np.int64).tolist()
    truncated=len(encoded.ids)>50
    matches=(observed==encoded.ids if not truncated else
             observed[:-1]==encoded.ids[:49] and observed[-1]==tokenizer.token_to_id('[SEP]'))
    mask_ok=bool(np.all(np.isin(attention,[0,1])) and
                 np.all(attention[:int(valid.sum())]==1) and
                 np.all(attention[int(valid.sum()):]==0))
    input_ok=bool(np.isfinite(ids).all() and np.all(ids==np.floor(ids)) and
                  np.all((ids>=0)&(ids<tokenizer.get_vocab_size())) and
                  np.all(ids[~valid]==0) and
                  np.all(np.isin(types,[0,1])) and
                  observed[0]==tokenizer.token_to_id('[CLS]') and
                  observed[-1]==tokenizer.token_to_id('[SEP]'))
    return matches,truncated,mask_ok,input_ok


def profile(samples,tokenizer):
    result={'samples':len(samples),'tokenizer_exact_or_truncated_prefix_match':0,
            'raw_text_over_50_tokens':0,'text_padding_mask_ok':0,
            'token_id_and_special_token_ok':0,'audio_all_zero':0,'vision_all_zero':0,
            'audio_interior_zero':0,'vision_interior_zero':0,
            'audio_trailing_zero_rows':0,'vision_trailing_zero_rows':0}
    problems=[]
    for sample in samples:
        sid=str(sample['id'])
        ok,truncated,mask_ok,input_ok=lexical_check(sample['text_bert'],sample['raw_text'],tokenizer)
        result['tokenizer_exact_or_truncated_prefix_match']+=ok
        result['raw_text_over_50_tokens']+=truncated
        result['text_padding_mask_ok']+=mask_ok
        result['token_id_and_special_token_ok']+=input_ok
        if not all((ok,mask_ok,input_ok)):
            problems.append({'sample_id':sid,'tokenizer_match':ok,
                             'text_mask_ok':mask_ok,'token_id_ok':input_ok})
        for modality in ('audio','vision'):
            active=np.any(np.asarray(sample[modality])!=0,axis=1)
            result[modality+'_all_zero']+=not active.any()
            if active.any():
                first,last=np.flatnonzero(active)[[0,-1]]
                result[modality+'_interior_zero']+=bool((~active[first:last+1]).any())
                result[modality+'_trailing_zero_rows']+=int((~active[last+1:]).sum())
    return result,problems


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    tokenizer=Tokenizer.from_file(str(a.tokenizer.resolve()))
    expected={'[PAD]':0,'[UNK]':100,'[CLS]':101,'[SEP]':102,'[MASK]':103}
    specials={name:tokenizer.token_to_id(name) for name in expected}
    if specials!=expected:raise ValueError(f'Unexpected tokenizer special IDs: {specials}')
    with next(a.data_root.resolve().glob('附件2-*/aligned_50.pkl')).open('rb') as f:
        blocks=pickle.load(f)
    result={'representation':'aligned_50 only','tokenizer_vocab_size':tokenizer.get_vocab_size(),
            'special_ids':specials,'splits':{},'problem_counts':{},
            'mask_semantics':{
                'text_padding':'text_bert attention mask: supplied valid token positions',
                'audio_vision_numeric_observation':'nonzero feature row; zero cause is not identifiable from aligned_50 alone',
                'audio_vision_padding':'not explicitly supplied; trailing zero rows are candidates, not verified padding truth',
                'known_injected_missing':'boolean mask recorded by the simulation, separate from original zeros'}}
    for split in ('train','valid','test'):
        b=blocks[split]
        n=len(b['id'])
        samples=({'id':b['id'][i],'raw_text':b['raw_text'][i],
                  'text_bert':b['text_bert'][i],
                  'audio':b['audio'][i],'vision':b['vision'][i]} for i in range(n))
        counts,problems=profile(list(samples),tokenizer)
        result['splits'][split]=counts
        result['problem_counts'][split]=len(problems)
        if problems:result.setdefault('first_problems',{})[split]=problems[:10]
    root=next(a.data_root.resolve().glob('附件3-*'))
    paths=sorted(root.rglob('对齐版本/*.pkl'))
    if len(paths)!=30:raise ValueError('Expected 30 aligned A3 samples')
    samples=[]
    for path in paths:
        item=load_small(path)
        if set(item)!={'test'}:raise ValueError('Unexpected A3 wrapper')
        normalized=normalize(item['test'],path.stem,'A3')
        normalized['id']=path.stem
        normalized['raw_text']=None
        samples.append(normalized)
    # A3 has no raw_text. Check ID space and masks without claiming a retokenization match.
    a3={'samples':len(samples),'token_id_and_special_token_ok':0,'text_padding_mask_ok':0,
        'audio_all_zero':0,'vision_all_zero':0,'audio_interior_zero':0,'vision_interior_zero':0}
    for sample in samples:
        ids=np.asarray(sample['text_bert'][0]);attention=np.asarray(sample['text_bert'][1]);types=np.asarray(sample['text_bert'][2])
        valid=attention!=0;observed=ids[valid]
        a3['text_padding_mask_ok']+=bool(np.all(np.isin(attention,[0,1])) and np.all(attention[:valid.sum()]==1)
                                         and np.all(attention[valid.sum():]==0))
        a3['token_id_and_special_token_ok']+=bool(np.all(ids==np.floor(ids)) and
            np.all((ids>=0)&(ids<tokenizer.get_vocab_size())) and np.all(ids[~valid]==0) and
            np.all(np.isin(types,[0,1])) and observed[0]==101 and observed[-1]==102)
        for modality in ('audio','vision'):
            active=np.any(sample[modality]!=0,axis=1)
            a3[modality+'_all_zero']+=not active.any()
            if active.any():
                first,last=np.flatnonzero(active)[[0,-1]]
                a3[modality+'_interior_zero']+=bool((~active[first:last+1]).any())
    result['splits']['A3']=a3
    result['A3_no_raw_text']='Cannot retokenize A3; validate numeric token ID domain, special IDs and attention mask only.'
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
