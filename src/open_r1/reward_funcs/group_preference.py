import re
from collections import defaultdict

def completion_to_mode(completions, group_prefix):
    completions = [completion[0]["content"] for completion in completions]
    modes = []
    for c in completions:
        if group_prefix is None:
            c_mode = "mode"
        else:
            for i,prefix in enumerate(group_prefix):
                if c.startswith(prefix):
                    break
            c_mode = "mode_{}".format(i)
        modes.append(c_mode)
    return modes

def group_pref_win_probability(completions, solution, **kwargs):
    """group preference based on win probability"""
    assert 'accuracy_reward' in kwargs, "the group preference should be based on accuracy"
    assert "group_prefix" in kwargs and kwargs['group_prefix'] is not None, "the group preference should be based on group prefix"
    assert len(kwargs['group_prefix']) == 2, "current function only supports two groups"
    accuracy_reward = kwargs['accuracy_reward']
    contents = [completion[0]["content"] for completion in completions]
    modes = completion_to_mode(completions, kwargs['group_prefix'])

    # compute the win probability by enumeration
    def win_probability(modes, accuracy_reward):
        mode2acc = defaultdict(list)
        for mode, acc in zip(modes, accuracy_reward):
            mode2acc[mode].append(acc)
        win_prob0 = 0
        win_prob1 = 0
        factor = len(mode2acc['mode_0']) * len(mode2acc['mode_1'])
        for r0 in mode2acc['mode_0']:
            for r1 in mode2acc["mode_1"]:
                if r0 > r1:
                    win_prob0 += 1
                elif r1 > r0:
                    win_prob1 += 1
        return {"mode_0": win_prob0/factor, "mode_1": win_prob1/factor}

    mode2winprob = win_probability(modes, accuracy_reward)
    rewards = []
    for mode, acc in zip(modes, accuracy_reward):
        r = 0
        if acc == 1:
            r = mode2winprob[mode]
        rewards.append(r)
    return rewards