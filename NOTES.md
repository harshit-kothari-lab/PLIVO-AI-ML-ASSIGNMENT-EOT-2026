# NOTES

1. **Signal used.** For each silence I read only `audio[0:pause_start]` and
   summarise the last ~1.5 s: falling pitch and pitch dropping below the turn's
   own baseline (strongest cue, especially Hindi), the shape of the energy
   offset, final-syllable lengthening and the F0 movement inside it, trailing
   voicing, creak, silence history measured from the audio, and coarse position.
2. Pitch is the money cue: before a true end-of-turn the speaker's F0 falls and
   settles ~2–5 semitones under their turn mean; before a hold it stays level or
   rises. In Hindi that gap is ~3 st and clean; in English it is weak and partly
   reversed, which is why English is the harder language.
3. The scorer only counts a false cutoff when the hold outlasts the chosen
   delay, so the discrimination worth paying for is EOT vs *long* hold — I
   re-measured every feature on that subset, and offset shape and tail-energy
   curvature separate far better there than they do overall.
4. **Causality** is structural: frame-local operators (`librosa` yin/rms/stft,
   never pyin's Viterbi over the whole file) computed once, then sliced to
   frames ending before `pause_start`; the pyin tail runs on a truncated
   pre-pause slice; `pause_end` is never read, and pause history comes from
   silences I detect in the audio rather than from the labels file.
5. Model is a small ensemble (two HistGradientBoosting + a scaled logistic
   regression) over 55 features, blended 2:1 towards the trees.
6. **What failed:** per-language models, isotonic calibration, and cost-aware
   hold weighting — the last one gained ~100 ms on my earlier 35-feature set
   and then *lost* ground once the features improved, so it was patching a weak
   feature set rather than adding signal.
7. **The delay metric is noisy.** Re-running the same config over eight fold
   shuffles gives English 1192 ± 34 ms while AUC holds at .688 ± .009, so I
   selected on AUC and report delay as a band; an earlier 1145 ms was a lucky
   draw, not a real gain.
8. English stays around 1200 ms because its turn-final prosody is genuinely
   ambiguous — many holds are long and sound complete — and pure acoustics has
   a low ceiling there without the words.
9. **In-sample scores are a trap** (0.997 AUC / 145 ms); the delivered
   `predictions_*.csv` are out-of-fold, and the shipped `predict.py` carries the
   all-data weights inline so it needs no companion file.
10. **One more day:** train a pairwise ranking objective restricted to
    EOT-vs-long-hold pairs (aligned to the actual cost model), mine more long
    holds, add phone-level cues (final vowel vs cut-off consonant), and try a
    small from-scratch GRU over the trailing 2 s contour.
