
unimotifcomparator motif ./bamm ./pif4.meme \
--model1-type bamm \
--model2-type pwm \
--motif-perm 500 \
--motif-metric co


unimotifcomparator motif ./gata2 ./gata4 \
--model1-type bamm \
--model2-type bamm \
--motif-perm 1000 \
--motif-metric cj


unimotifcomparator motif ./sitega_stat6.mat ./pif4.meme \
--model1-type sitega \
--model2-type pwm \
--motif-perm 500 \
--motif-metric co

unimotifcomparator tomtom-like ./sitega_gata2.mat ./pif4.meme \
--model1-type sitega \
--model2-type pwm \
--metric ed \
--permutations 500

unimotifcomparator tomtom-like ./pif4.meme ./pif4.meme \
--model1-type pwm \
--model2-type pwm \
--metric ed \
--permutations 10000 \
--permute-rows

unimotifcomparator tomtom-like ./sitega_stat6.mat ./pif4.meme \
--model1-type sitega \
--model2-type pwm \
--metric pcc \
--permutations 1000 \
--pfm-mode 

unimotifcomparator sequence ./sitega.mat ./pif4.meme \
--model1-type sitega \
--model2-type pwm \
--metric cj \
--perm 1000

unimotifcomparator tomtom-like ./sitega_stat6.mat ./sitega_gata2.mat \
--model1-type sitega \
--model2-type sitega \
--metric pcc \
--permutations 10000 \
--permute-rows \
--pfm-mode

uv run unimotifcomparator tomtom-like ./sitega_stat6.mat ./sitega_gata2.mat \
--model1-type sitega \
--model2-type sitega \
--metric ed \
--permutations 10000 \
--permute-rows \
--pfm-mode

uv run unimotifcomparator tomtom-like ./sitega_stat6.mat ./sitega_stat6.mat \
--model1-type sitega \
--model2-type sitega \
--metric ed \
--permutations 10000 \
--permute-rows \
--pfm-mode


unimotifcomparator tomtom-like ./gata2.meme ./sitega_gata2.mat \
--model1-type pwm \
--model2-type sitega \
--metric ed \
--permutations 1000 \
--pfm-mode

unimotifcomparator tomtom-like ./gata2 ./gata4 \
--model1-type bamm \
--model2-type bamm \
--permutations 1000 \
--metric ed -v 


unimotifcomparator score ./scores_1.fasta ./scores_2.fasta

uv run unimotifcomparator motali ./gata2.meme ./sitega_gata2.mat \
--model1-type pwm \
--model2-type sitega
