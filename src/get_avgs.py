import srsly 
import numpy as np
import fire 


def get_avgs(path):
    res = srsly.read_jsonl(path)
    times = []
    corrects = []
    parses = []
    for r in res:
        times.append(r['time'])
        corrects.append(r['correct'])
        parses.append(r['parses'])
    
    print('Mean Accuracy ', np.mean(corrects))
    print('Mean Time ', np.mean(times))
    return {'acc': round(100 * np.mean(corrects),2),
            'time': round(np.mean(times), 2), 
            'parses': round(100 * np.mean(parses), 2)
            }

if __name__ == '__main__':
    fire.Fire(get_avgs)