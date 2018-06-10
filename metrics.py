import numpy as np

from spectrogram import get_power

def compute_psnr(x_hr, x_pr):
  # max_hr = np.max(np.abs(x_hr))
  max_hr = np.sqrt(np.mean(x_hr**2))
  # max_pr = np.max(np.abs(x_pr))
  # x_pr = x_pr / max_pr * max_hr
  loss = np.mean((x_pr - x_hr)**2)
  psnr = 20 * np.log10(max_hr / np.sqrt(loss) + 1e-8)
  # if len(x_hr) == 8000:
  #   print np.sqrt(loss), max_hr, np.max(x_pr)
  #   print ' '.join(['%.3f' % f for f in x_hr[5000:5050]])
  #   print ' '.join(['%.3f' % f for f in x_pr[5000:5050]])
  #   # print ' '.join(['%.3f' % f for f in x_hr[40000:40050]])
  #   # print ' '.join(['%.3f' % f for f in x_pr[40000:40050]])

  return psnr

WIN_SIZE=2048
def compute_segsnr(x_hr, x_pr):
  n_windows = len(x_hr) // WIN_SIZE
  x_hr_wins = [x_hr[i*WIN_SIZE:(i+1)*WIN_SIZE] for i in range(n_windows)]
  x_pr_wins = [x_pr[i*WIN_SIZE:(i+1)*WIN_SIZE] for i in range(n_windows)]

  x_wins = [ (x_hr_win, x_pr_win) for x_hr_win, x_pr_win in zip(x_hr_wins, x_pr_wins) ]

  psnrs = [compute_psnr(x_hr_win, x_pr_win) for (x_hr_win, x_pr_win) in x_wins]

  # for psnr, (x_hr_win, x_pr_win) in zip(psnrs, x_wins):
  #   max_hr = np.sqrt(np.mean(x_hr_win**2))
  #   loss = np.mean((x_pr_win - x_hr_win)**2)
  #   psnr = 20 * np.log10(max_hr / np.sqrt(loss) + 1e-8)
  #   print psnr, loss, np.mean((x_hr_win - x_pr_win)**2), max_hr, max_hr / np.sqrt(loss), np.mean(np.abs(x_hr_win)), np.mean(np.abs(x_pr_win))
  # # print [(psnr, np.mean(np.abs(x_hr_win))) for psnr, (x_hr_win, _) in zip(psnrs, x_wins)]
  return np.mean(np.array(psnrs))

def compute_log_distortion(x_hr, x_pr):
  S1 = get_power(x_hr) # (n_frames, n_freq)
  S2 = get_power(x_pr)

  lsd = np.mean(np.sqrt(np.mean((S1 - S2)**2 + 1e-8, axis=1)), axis=0)
  return min(lsd, 10.)