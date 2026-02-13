mimosa motif ./bamm ./pif4.meme `
--model1-type bamm `
--model2-type pwm `
--permutations 500 `
--metric co

mimosa motif ./gata2.ihbcp ./gata4.ihbcp `
--model1-type bamm `
--model2-type bamm `
--permutations 1000 `
--metric cj

mimosa motif ./sitega_stat6.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--permutations 500 `
--metric co

mimosa tomtom-like ./sitega_gata2.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--metric ed `
--permutations 500

mimosa tomtom-like ./pif4.meme ./pif4.meme `
--model1-type pwm `
--model2-type pwm `
--metric ed `
--permutations 10000 `
--permute-rows

mimosa tomtom-like ./sitega_stat6.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--metric pcc `
--permutations 1000 `
--pfm-mode

mimosa sequence ./sitega.mat ./pif4.meme `
--model1-type sitega `
--model2-type pwm `
--metric cj `
--permutations 1000

mimosa tomtom-like ./sitega_stat6.mat ./sitega_gata2.mat `
--model1-type sitega `
--model2-type sitega `
--metric pcc `
--permutations 10000 `
--permute-rows `
--pfm-mode

mimosa tomtom-like ./sitega_stat6.mat ./sitega_gata2.mat `
--model1-type sitega `
--model2-type sitega `
--metric ed `
--permutations 10000 `
--permute-rows `
--pfm-mode

mimosa tomtom-like ./sitega_stat6.mat ./sitega_stat6.mat `
--model1-type sitega `
--model2-type sitega `
--metric ed `
--permutations 10000 `
--permute-rows `
--pfm-mode

mimosa tomtom-like ./gata2.meme ./sitega_gata2.mat `
--model1-type pwm `
--model2-type sitega `
--metric ed `
--permutations 1000 `
--pfm-mode

mimosa tomtom-like ./gata2.ihbcp ./gata4.ihbcp `
--model1-type bamm `
--model2-type bamm `
--permutations 1000 `
--metric ed -v

mimosa score ./scores_1.fasta ./scores_2.fasta

mimosa motali ./gata2.meme ./sitega_gata2.mat `
--model1-type pwm `
--model2-type sitega
