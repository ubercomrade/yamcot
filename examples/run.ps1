yamcot motif ./bamm ./pif4.meme `
--model1-type bamm `
--model2-type pwm `
--perm 500 `
--metric co

yamcot motif ./gata2 ./gata4 `
--model1-type bamm `
--model2-type bamm `
--perm 1000 `
--metric cj

yamcot motif ./sitega_stat6.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--perm 500 `
--metric co

yamcot tomtom-like ./sitega_gata2.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--metric ed `
--permutations 500

yamcot tomtom-like ./pif4.meme ./pif4.meme `
--model1-type pwm `
--model2-type pwm `
--metric ed `
--permutations 10000 `
--permute-rows

yamcot tomtom-like ./sitega_stat6.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--metric pcc `
--permutations 1000 `
--pfm-mode 

yamcot sequence ./sitega.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--metric cj `
--perm 1000

yamcot tomtom-like ./sitega_stat6.mat ./sitega_gata2.mat `
--model1-type sitega `
--model2-type sitega `
--metric pcc `
--permutations 10000 `
--permute-rows `
--pfm-mode

yamcot tomtom-like ./sitega_stat6.mat ./sitega_gata2.mat `
--model1-type sitega `
--model2-type sitega `
--metric ed `
--permutations 10000 `
--permute-rows `
--pfm-mode

yamcot tomtom-like ./sitega_stat6.mat ./sitega_stat6.mat `
--model1-type sitega `
--model2-type sitega `
--metric ed `
--permutations 10000 `
--permute-rows `
--pfm-mode

yamcot tomtom-like ./gata2.meme ./sitega_gata2.mat `
--model1-type pwm `
--model2-type sitega `
--metric ed `
--permutations 1000 `
--pfm-mode

yamcot tomtom-like ./gata2 ./gata4 `
--model1-type bamm `
--model2-type bamm `
--permutations 1000 `
--metric ed -v 

yamcot score ./scores_1.fasta ./scores_2.fasta

yamcot motali ./gata2.meme ./sitega_gata2.mat `
--model1-type pwm `
--model2-type sitega
