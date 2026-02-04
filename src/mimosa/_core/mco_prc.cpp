#define _CRT_SECURE_NO_WARNINGS

#include  <stdio.h>
#include  <stdlib.h>
#include  <string.h>
#include  <math.h>
#include  <time.h>
#include  <ctype.h>

#define Min(a,b) ((a)>(b))? (b):(a);
#define Max(a,b) ((a)>(b))? (a):(b);
#define SEQLEN 12000
#define MATLEN 50 //max matrix length
#define SPACLEN 100 //max spacer length
#define ARGLEN 300 //max argv length
#define OLIGNUM 4// di 16 mono 4
#define DIM 100

struct qbs {
	double err;//ERR score
//	int m0;//for given ERR no. of better sites of this model
//	int m1;//for given ERR no. of better sites of other model
	int mod;// model 0 or 1
};
int compare_qbs(const void* X1, const void* X2)//decrease
{
	struct qbs* S1 = (struct qbs*)X1;
	struct qbs* S2 = (struct qbs*)X2;
	if (S1->err - S2->err > 0)return -1;
	if (S1->err - S2->err < 0)return 1;
	return 0;
}
int StrNStr(char* str, char c, int n)
{
	int i, len = (int)strlen(str);
	int k = 0;
	for (i = 0; i < len; i++)
	{
		if (str[i] == c)
		{
			k++;
			if (k == n)return i;
		}
	}
	return -1;
}
int StrEndNStr(char* str, char c, int n)
{
	int i, len = (int)strlen(str);
	int k = 0;
	for (i = len - 1; i >= 0; i--)
	{
		if (str[i] == c)
		{
			k++;
			if (k == n)return i;
		}
	}
	return -1;
}
char* TransStr(char* d)
{
	int i, c, lens;
	lens = strlen(d);
	for (i = 0; i < lens; i++)
	{
		c = int(d[i]);
		if (c < 97) d[i] = char(c + 32);
		//else break;
	}
	return(d);
}
char* TransStrBack(char* d)
{
	int i, c, lens;
	lens = strlen(d);
	for (i = 0; i < lens; i++)
	{
		c = int(d[i]);
		if (c >= 97) d[i] = char(c - 32);
		//else break;
	}
	return(d);
}
void DelChar(char* str, char c)
{
	int i, lens, size;

	size = 0;
	lens = (int)strlen(str);
	for (i = 0; i < lens; i++)
	{
		if (str[i] != c)str[size++] = str[i];
	}
	str[size] = '\0';
}
int ComplStr(char* d)
{
	char* d1;
	int i, len;
	len = strlen(d);
	d1 = new char[len + 1];
	if (d1 == NULL)
	{
		fprintf(stderr, "Cmpl: Out of memory...");
		return 0;
	}
	strcpy(d1, d);
	//	memset(d,0,sizeof(d));
	for (i = 0; i < len; i++)
	{
		switch (d1[len - i - 1])
		{
		case 'a': { d[i] = 't'; break; }
		case 't': { d[i] = 'a'; break; }
		case 'c': { d[i] = 'g'; break; }
		case 'g': { d[i] = 'c'; break; }
		case 'A': { d[i] = 'T'; break; }
		case 'T': { d[i] = 'A'; break; }
		case 'C': { d[i] = 'G'; break; }
		case 'G': { d[i] = 'C'; break; }
		case 'N': { d[i] = 'N'; break; }
		case 'n': { d[i] = 'n'; break; }
		default: d[i] = 'n';
		}
	}
	delete[] d1;
	return 1;
}
int UnderStolStr(char* str, int nstol, char* ret, size_t size, char sep)
{
	memset(ret, '\0', size);
	int p1, p2, len;
	if (nstol == 0)
	{
		p2 = StrNStr(str, sep, 1);
		if (p2 == -1)p2 = strlen(str);
		strncpy(ret, str, p2);
		ret[p2] = '\0';
		return 1;
	}
	else
	{
		p1 = StrNStr(str, sep, nstol);
		p2 = StrNStr(str, sep, nstol + 1);
		if (p2 == -1)
		{
			p2 = strlen(str);
		}
		if (p1 == -1 || p2 == -1) return -1;
		len = p2 - p1 - 1;
		strncpy(ret, &str[p1 + 1], len);
		ret[len] = '\0';
		return 1;
	}
}
#include "fasta_to_plain.h"
#include "pfm_to_pwm.h"
//#include "pwm_rec.h"
//#include "pfm_similarity.h"

struct due {
	double buf;
	int sta;
	int end;
	int num;
	void get_copy(due* a);
	//	void print_all(void);
};
void due::get_copy(due* a)
{
	a->num = num;
	a->sta = sta;
	a->buf = buf;
	a->end = end;
};

//set of dinucleotides
struct city {
	char site[300];
	int size;
	int len;
	double min;
	double raz;
	struct due tot[DIM];
	void get_copy(city* a);
	void sort_all(void);
	int get_file(char* file);
	//void city::fprint_tab(char *file);
}sta;
int city::get_file(char* file)
{
	FILE* in;
	if ((in = fopen(file, "rt")) == NULL)
	{
		printf("Input file %s can't be opened!", file);
		return -1;
	}
	char d[300];
	fgets(d, sizeof(d), in);
	DelChar(d,'\n');
	strcpy(site, d);
	fgets(d, sizeof(d), in);
	size = atoi(d);
	fgets(d, sizeof(d), in);
	len = atoi(d);
	fgets(d, sizeof(d), in);
	min = atof(d);
	fgets(d, sizeof(d), in);
	raz = atof(d);
	char sep = '\t', s[30];
	int i, test;
	for (i = 0; i < size; i++)
	{
		fgets(d, sizeof(d), in);
		tot[i].sta = atoi(d);
		test = UnderStolStr(d, 1, s, sizeof(s), sep);
		if (test == -1) { printf("Wrong format %s\n", d); return(-1); }
		tot[i].end = atoi(s);
		test = UnderStolStr(d, 2, s, sizeof(s), sep);
		if (test == -1) { printf("Wrong format %s\n", d); return(-1); }
		tot[i].buf = atof(s);
		test = UnderStolStr(d, 3, s, sizeof(s), sep);
		if (test == -1) { printf("Wrong format %s\n", d); return(-1); }
		tot[i].num = atoi(s);
	}
	fclose(in);
	return 1;
}
void city::get_copy(city* a)
{
	strcpy(a->site, site);
	a->size = size;
	a->min = min;
	a->len = len;
	a->raz = raz;
	int i;
	for (i = 0; i < size; i++)
	{
		tot[i].get_copy(&a->tot[i]);
	}
}
int compare_due(const void* X1, const void* X2)
{
	struct due* S1 = (struct due*)X1;
	struct due* S2 = (struct due*)X2;
	if (S1->sta - S2->sta > 0)return 1;
	if (S1->sta - S2->sta < 0)return -1;
	if (S1->end - S2->end > 0)return 1;
	if (S1->end - S2->end < 0)return -1;
	if (S1->num - S2->num > 0)return 1;
	if (S1->num - S2->num < 0)return -1;
	return 0;
}
void city::sort_all(void)
{
	qsort((void*)tot, size, sizeof(tot[0]), compare_due);
}
int IdeLet(char c)
{
	int ret;
	switch (c) {
	case 'a': ret = 0; break;
	case 'c': ret = 1; break;
	case 'g': ret = 2; break;
	case 't': ret = 3; break;
	case 'n': ret = -1; break;
	default: ret = -2;
	}
	return(ret);
}
// ras4et 4asot oligonukleotidov po stroke (zdes' - nukleotidov)
void GetSostPro(char* d, int word, int* sost)
{
	int i, j, k, i_sost, let;
	char letter[] = "acgt";
	int ten[6] = { 1, 4, 16, 64, 256, 1024 };
	int lens = strlen(d);
	int size = 1;
	for (k = 0; k < word; k++)size *= 4;
	for (i = 0; i < size; i++)sost[i] = 0;
	for (i = 0; i < lens - word + 1; i++)
	{
		i_sost = 0;
		let = -1;
		for (j = word - 1; j >= 0; j--)
		{
			for (k = 0; k < 4; k++)
			{
				if (d[i + j] == letter[k]) { let = k; break; }
			}
			i_sost += ten[word - 1 - j] * let;
		}
		sost[i] = i_sost;
	}
}
void PWMScore(double **pwm, double& min, double& raz, int len1)
{
	int i, j;
	for (i = 0; i < len1; i++)
	{
		double pwmmin = 100;
		double pwmmax = -100;
		for (j = 0; j < OLIGNUM; j++)
		{
			if (pwm[i][j] < pwmmin)pwmmin = pwm[i][j];
			if (pwm[i][j] > pwmmax)pwmmax = pwm[i][j];
		}
		raz += pwmmax;
		min += pwmmin;
	}
	raz -= min;
}
int PWM_SGA_rec_real(double ***pwm, double min[2], double raz[2], city sta[2], int model_type[2], int nthr_dist[2], double** thr_all, double** fpr_all, char*** seq, int olen[2], int nseq,
	int shift, int length_fasta_max, char *file_hist, char* file_prc, int yes_out_hist, int yes_out_prc,double *auprc_final_all, double *auprc_final_over,
	double &auprc_final_over1,double &auprc_final_all1,double *shift_final_all,double *shift_final_over,char *strand_final_all,char *strand_final_over)
{
	int i, j, k, n, m;
	int compl1, kmin, kmax;
	int cod[MATLEN];
	char d[MATLEN];
	int word = 1;
	int nthr_dist1[2], olen1[2];
	double thr_cr[2];
	int nthr_dist_two = nthr_dist[0] + nthr_dist[1];
	if (olen[0] < olen[1]) { kmin = 0; kmax = 1; }
	else { kmin = 1; kmax = 0; }
	double wshift_ov[3];
	int n_shift[3];
	for (i = 0; i < 3; i++) n_shift[i] = 2 * shift;
	int j_ov1[3], j_ov2[3];
	int kpairs[3][2];
	kpairs[0][0] = kpairs[1][0] = kpairs[1][1] = kmax;
	kpairs[0][1] = kpairs[2][0] = kpairs[2][1] = kmin;
	double** wshift;
	wshift = new double* [3];
	if (wshift == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
	for(i = 0; i < 3; i++)
	{
		int kk[2];
		int chet[2];
		for (k = 0; k < 2; k++)
		{
			kk[k] = kpairs[i][k];
			if (olen[kk[k]] % 2 == 0)chet[k] = 1;//4etno
			else chet[k] = 0;//ne4etno
		}
		if (chet[0] == chet[1])n_shift[i]++;
		if (chet[0] == chet[1])// -1 0 +1
		{
			if (chet[0] == 0)wshift_ov[i] = (double)(olen[kk[0]] - 1) / 2;
			else wshift_ov[i] = (double)olen[kk[0]] / 2;
		}
		else //-0.5 +0.5
		{
			if (chet[kk[0]] == 1)wshift_ov[i] = (double)olen[kk[0]] / 2 - 0.5;
			else wshift_ov[i] = (double)(olen[kk[0]] - 1) / 2 + 0.5;
		}
		//shift = n_shift / 2;
		wshift[i] = new double[n_shift[i]];
		if (wshift[i] == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
		if (chet[0] == chet[1])
		{
			wshift[i][0] = -shift;
		}
		else
		{
			wshift[i][0] = 0.5 - shift;
		}
		for (k = 1; k < n_shift[i]; k++)wshift[i][k] = wshift[i][k - 1] + 1;
		for (k = 0; k < n_shift[i]; k++)
		{
			if (wshift[i][k] == -wshift_ov[i])
			{
				j_ov1[i] = k; continue;
			}
			if (wshift[i][k] == wshift_ov[i])
			{
				j_ov2[i] = k; break;
			}
		}
	}
	double*** auprc;
	auprc = new double** [3];
	for (j = 0; j < 3; j++)
	{
		auprc[j] = new double* [2];
		if (auprc[j] == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
		for (k = 0; k < 2; k++)
		{
			auprc[j][k] = new double[n_shift[j]];
			if (auprc[j][k] == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
		}
	}
	for (j = 0; j < 3; j++)for (k = 0; k < 2; k++)for (m = 0; m < n_shift[j]; m++)auprc[j][k][m] = 0;
	double half_win[2];//double shift of center relative to start
	for (k = 0; k < 2; k++)half_win[k] = ((double)olen[k] - 1) / 2;
	int* inx_self[2];
	for (k = 0; k < 2; k++)
	{
		inx_self[k] = new int[nthr_dist[k]];
		if (inx_self[k] == NULL) { fprintf(stderr, "Inx_self Out of memory..."); return -1; }
	}
	for (k = 0; k < 2; k++)for (i = 0; i < nthr_dist[k]; i++)inx_self[k][i] = nthr_dist_two;
	qbs* errs;
	errs = new qbs[nthr_dist_two];
	if (errs == NULL) { fprintf(stderr, "Inx_self Out of memory..."); return -1; }
	j = 0;
	for (k = 0; k < 2; k++)
	{
		for (i = 0; i < nthr_dist[k]; i++)
		{
			errs[j].err = fpr_all[k][i];
		//	errs[j].m0 = i;
			errs[j].mod = k;
			j++;
		}
	}
	qsort(errs, nthr_dist_two, sizeof(errs[0]), compare_qbs);
	{
		int cou[2];
		for (k = 0; k < 2; k++)cou[k] = 0;
		for (k = 0; k < nthr_dist_two; k++)
		{
			int model = errs[k].mod;
			//int model1 = 1 - errs[k].mod;
			inx_self[model][cou[model]] = k;
			//inx_cross[model1][cou[model1]] = k;
			cou[model]++;
		}
	}
	int **** tp;
	double ** fp;
	tp = new int*** [3];
	if (tp == NULL) { fprintf(stderr, "TP Out of memory..."); return -1; }

	for (j = 0; j < 3; j++)
	{
	    tp[j] = new int** [2];
	    if (tp[j] == NULL) { puts("TP Out of memory..."); return -1; }

	    for (k = 0; k < 2; k++)
	    {
	        tp[j][k] = new int* [n_shift[j]];
	        if (tp[j][k] == NULL) { puts("TP Out of memory..."); return -1; }  // Исправлено

	        for (i = 0; i < n_shift[j]; i++)
	        {
	            tp[j][k][i] = new int[nthr_dist_two];
	            if (tp[j][k][i] == NULL) { fprintf(stderr, "TP Out of memory..."); return -1; }
	        }
	    }
	}
	for (j = 0; j < 3; j++)for (k = 0; k < 2; k++)for (i = 0; i < n_shift[j]; i++)for (m = 0; m < nthr_dist_two; m++)tp[j][k][i][m] = 0;
	fp = new double* [3];
	if (fp == NULL) { fprintf(stderr, "FP Out of memory..."); return -1; }
	for (j = 0; j < 3; j++)
	{
		fp[j] = new double[nthr_dist_two];
		if (fp[j] == NULL) { fprintf(stderr, "FP Out of memory..."); return -1; }
	}
	for (j = 0; j < 3; j++)for (m = 0; m < nthr_dist_two; m++)fp[j][m] = 0;
	int*** tp_tot;
	double fp_tot[3] = { 0,0,0 };
	tp_tot = new int** [3];
	if (tp_tot == NULL) { fprintf(stderr, "TP_tot Out of memory..."); return -1; }
	for (j = 0; j < 3; j++)
	{
		tp_tot[j] = new int* [2];
		if (tp_tot[j] == NULL) { puts("TP_tot Out of memory..."); return -1; }
		for (k = 0; k < 2; k++)
		{
			tp_tot[j][k] = new int[n_shift[j]];
			if (tp_tot[j][k] == NULL) { puts("TP_tot Out of memory..."); return -1; }
		}
	}
	for (j = 0; j < 3; j++) for (k = 0; k < 2; k++)for (i = 0; i < n_shift[j]; i++)tp_tot[j][k][i] = 0;
	int*** err_inx;
	err_inx = new int** [2];
	if (err_inx == NULL) { puts("Err_inx Out of memory..."); return -1; }
	for (k = 0; k < 2; k++)
	{
		err_inx[k] = new int* [2];
		if (err_inx[k] == NULL) { puts("Err_inx Out of memory..."); return -1; }
		for (i = 0; i < 2; i++)
		{
			err_inx[k][i] = new int[length_fasta_max];
			if (err_inx[k][i] == NULL) { fprintf(stderr, "Err_inx Out of memory..."); return -1; }
		}
	}
	for (k = 0; k < 2; k++)for (i = 0; i < 2; i++)for (j = 0; j < length_fasta_max; j++)err_inx[k][i][j] = nthr_dist_two;
	for (n = 0; n < 2; n++)
	{
		nthr_dist1[n] = nthr_dist[n] - 1;
		olen1[n] = olen[n] - 1;
		thr_cr[n] = thr_all[n][nthr_dist1[n]];
	}
	for (n = 0; n < nseq; n++)
	{
		//if ((n + 1) % 100 == 0)printf("\b\b\b\b\b\b\b%7d", n + 1);
		int len_pro1 = strlen(seq[0][n]);
		for (k = 0; k < 2; k++)
		{
			int len21 = len_pro1 - olen[k];
		//	for (j = 0; j < 2; j++)for (i = 0; i <= len21; i++)err_inx[k][j][i] = nthr_dist_two;
			for (i = 0; i <= len21; i++)
			{
				int index = nthr_dist_two;
				double sco2 = 0;
				int gom = 0;
				for (compl1 = 0; compl1 < 2; compl1++)
				{
					int ista;
					if (compl1 == 0)ista = i;
					else ista = len21 - i;
					strncpy(d, &seq[compl1][n][ista], olen[k]);
					d[olen[k]] = '\0';
					if (strstr(d, "n") != NULL) { gom = 1; break; }
					double score = 0;
					if (model_type[k] == 0)
					{
						GetSostPro(d, word, cod);
						for (j = 0; j < olen[k]; j++)
						{
							score += pwm[k][j][cod[j]];
						}
						score -= min[k];
						score /= raz[k];
					}
					else
					{
						for (j = 0; j < sta[k].size; j++)
						{
							int rlenj = (sta[k].tot[j].end - sta[k].tot[j].sta + 1);
							double fm = 0;
							for (m = sta[k].tot[j].sta; m <= sta[k].tot[j].end; m++)
							{
								int cod = 4 * IdeLet(d[m]) + IdeLet(d[m + 1]);
								if (sta[k].tot[j].num == cod) { fm++; }
							}
							if (fm != 0)
							{
								fm /= rlenj;
								score += sta[k].tot[j].buf * fm;
							}
						}
						score -= sta[k].min;
						score /= sta[k].raz;
					}
					if (score > sco2)sco2 = score;
					if (gom == 0)
					{
						if (sco2 >= thr_cr[k])
						{
							if (sco2 >= thr_all[k][0])
							{
								index = 0;
								//break;
							}
							else
							{
								for (j = 1; j < nthr_dist[k]; j++)
								{
									if (sco2 >= thr_all[k][j] && sco2 < thr_all[k][j - 1])
									{
										index = j;
										break;
									}
								}
							}
						}
						//if (index == 0)break;
						err_inx[k][compl1][i] = index;
					}
				}
			}
		}
		int len21[2];
		for (k = 0; k < 2; k++)
		{
			len21[k] = len_pro1 - olen[k];
		}
		/*for (k = 0; k < 2; k++)
		{
			printf("%d\n",k);
			for (i = 0; i <= len21[k]; i++)
			{
				for (j = 0; j < 2; j++)if(err_inx[k][j][i]<nthr_dist_two)printf("%.1f_%d_%d ",half_win[j] + i, j, err_inx[k][j][i]);
				if ((i + 1) % 200 == 0)printf("\n");
			}
			printf("\n");
		}	*/
		for(j = 0; j < 3 ; j++)
		{
			int n_shift1 = n_shift[j] - 1;
			int j1 = kpairs[j][0];
			int j2 = kpairs[j][1];
			for (m = 0; m <= len21[j1]; m++)
			{
				{
					double cent_win1 = m + half_win[j1];
					int inx1[2];
					for(k = 0; k < 2 ; k++)inx1[k] = err_inx[j1][k][m];
					int mini1 = Min(inx1[0], inx1[1]);
					if (mini1 == nthr_dist_two)continue;
					for (i = 0; i <= len21[j2]; i++)
					{
						int inx2[2];
						for (k = 0; k < 2; k++)inx2[k] = err_inx[j2][k][i];
						int mini2 = Min(inx2[0], inx2[1]);
						if (mini2 == nthr_dist_two)continue;
						double cent_win2 = i + half_win[j2];
						double cent_dif = cent_win2 - cent_win1;
						int ori1, ori2;
						for (ori1 = 0; ori1 < 2; ori1++)
						{
							for (ori2 = 0; ori2 < 2; ori2++)
							{
								if (inx1[ori1] == nthr_dist_two || inx2[ori2] == nthr_dist_two)continue;
								int ori_type;
								if (ori1 == ori2)ori_type = 0;//Direct
								else ori_type = 1; //Invert or Evert
								int k1 = inx_self[j1][inx1[ori1]];
								int k2 = inx_self[j2][inx2[ori2]];
								if (cent_dif >= wshift[j][0] && cent_dif <= wshift[j][n_shift1])
								{
									int shift_pos;
									if (ori1 == 0)shift_pos = (int)(cent_win2 - cent_win1 - wshift[j][0]);//Direct12 Invert
									else shift_pos = (int)(cent_win1 - cent_win2 - wshift[j][0]);//Direct21 Evert
									tp_tot[j][ori_type][shift_pos]+=2;
									tp[j][ori_type][shift_pos][k1]++;
									tp[j][ori_type][shift_pos][k2]++;
								}
								else
								{
									double cent_diff = fabs(cent_dif);
									if(cent_diff <= n_shift[j])
									{
										fp_tot[j] += 2;
										fp[j][k1]++;
										fp[j][k2]++;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	FILE* out_hist;
	FILE* out_prc;
	out_hist = NULL;
	out_prc = NULL;
	if (yes_out_hist == 1)
	{
		if ((out_hist = fopen(file_hist, "wt")) == NULL)
		{
			printf("Output file can't be opened!\n");
			return -1;
		}
	}
	if (yes_out_prc == 1)
	{
		if ((out_prc = fopen(file_prc, "wt")) == NULL)
		{
			printf("Output file can't be opened!\n");
			return -1;
		}
	}
	for (j = 0; j < 3; j++)
	{
		if (yes_out_hist == 1)
		{
			if (j == 0)fprintf(out_hist, "Heterotypic\t");
			else fprintf(out_hist, "Homotypic\t");
			fprintf(out_hist, "Motif1\t%d\tMotif2\t%d\n", olen[kpairs[j][0]], olen[kpairs[j][1]]);
			for (i = 0; i < n_shift[j]; i++)
			{
				fprintf(out_hist, "\t%.1f", wshift[j][i]);
			}
			fprintf(out_hist, "\n");
		}
		{
			int del = n_shift[j] * 2;
			fp_tot[j] /= del;
			for (i = 0; i < nthr_dist_two; i++)fp[j][i] /= del;
		}
		/*printf("\n");
		for (k = 0; k < 2; k++)
		{
			printf("%d TP\t",k);
			for (m = 0; m < n_shift; m++)
			{
				printf("%.f ", tp_tot[k][m]);
			}
			printf("\t\t");
			printf("%d FP\t", k);
			//for (m = 0; m < n_shift; m++)
			{
				printf("%.f ", fp_tot);
			}
			printf("\n");
		}	*/
		for (k = 0; k < 2; k++)
		{
			if (yes_out_prc == 1)
			{
				if (j == 0)fprintf(out_prc, "Heterotypic\t");
				else fprintf(out_prc, "Homotypic\t");
				fprintf(out_prc, "Motif1\t%d\tMotif2\t%d\n", olen[kpairs[j][0]], olen[kpairs[j][1]]);
				if (k == 0)
				{
					fprintf(out_hist, "Direct ShortLong,LongShort");
				}
				else
				{
					fprintf(out_hist, "Evered Inverted");
				}
			}
			for (m = 0; m < n_shift[j]; m++)
			{
				if (yes_out_prc == 1)
				{
					if (k == 0)
					{
						if (wshift[j][m] < 0)fprintf(out_prc, "\tDirect ShortLong");
						else
						{
							if (wshift[j][m] > 0)fprintf(out_prc, "\tDirect LongShort");
							else fprintf(out_prc, "\tDirect Exact");
						}
					}
					else
					{
						if (wshift[j][m] < 0)fprintf(out_prc, "\tEverted");
						else
						{
							if (wshift[j][m] > 0)fprintf(out_prc, "\tInverted");
							else fprintf(out_prc, "\tReverse Exact");
						}
					}
					fprintf(out_prc, " %.1f\n", wshift[j][m]);
				}
				{
					int nthr_dist_two1 = nthr_dist_two - 1;
					double tpr_pred = 0, prec_pred = 1, tpr = 0;
					double dtp = 0, dfp = 0;
					int count_pr = 0, count_roc = 0;
					double tp_sum = 0, fp_sum = 0;
					if (yes_out_prc == 1)fprintf(out_prc, "%f\t%f\n", tpr_pred, prec_pred);
					for (i = 0; i < nthr_dist_two; i++)
					{
						dtp += (double)tp[j][k][m][i];
						dfp += fp[j][i];
						if (dtp > 0 && (i == nthr_dist_two1 || errs[i + 1].err != errs[i].err))
						{
							tp_sum += dtp;
							fp_sum += dfp;
							double prec_cur = tp_sum / (tp_sum + fp_sum);
							tpr = tp_sum / tp_tot[j][k][m];
							double prec_av = (prec_pred + prec_cur) / 2;
							double dauprc = dtp * prec_av / tp_tot[j][k][m];
							if (yes_out_prc == 1)fprintf(out_prc, "%f\t%f\n", tpr, prec_cur);
							prec_pred = prec_cur;
							tpr_pred = tpr;
							auprc[j][k][m] += dauprc;
							dtp = 0;
							dfp = 0;
							count_pr++;
						}
					}
				}
				if (yes_out_hist == 1)fprintf(out_hist, "\t%f", auprc[j][k][m]);
			}
			if (yes_out_hist == 1)fprintf(out_hist, "\n");
		}
	}
	if (yes_out_prc == 1)fclose(out_prc);
	if (yes_out_hist == 1)fclose(out_hist);
	double auprc_max[3][2] = { { -1,-1 },{ -1,-1 },{ -1,-1 } };
	double auprc_ov[3][2] = { { -1,-1 },{ -1,-1 },{ -1,-1 } };
	//int j_best[3][2] = { { 0,0 },{ 0,0 },{ 0,0 } };
	//int j_ov[3][2] = { { 0,0 },{ 0,0 },{ 0,0 } };
	char cepi[] = "+-";
	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < n_shift[j]; i++)
		{
			if (wshift[j][i] >= -wshift_ov[j] && wshift[j][i] <= wshift_ov[j])
			{
				for (k = 0; k < 2; k++)
				{
					if (auprc[j][k][i] > auprc_ov[j][k])
					{
						auprc_ov[j][k] = auprc[j][k][i];
						strand_final_over[j] = cepi[k];
						shift_final_over[j] = wshift[j][i];
						auprc_final_over[j] = auprc[j][k][i];
					}
				}
			}
		}
		for (i = 0; i < n_shift[j]; i++)
		{
			for (k = 0; k < 2; k++)
			{
				if (auprc[j][k][i] > auprc_max[j][k])
				{
					auprc_final_all[j] = auprc_max[j][k] = auprc[j][k][i];
					strand_final_all[j] = cepi[k];
					shift_final_all[j] = wshift[j][i];
				}
			}
		}
	}
	/*for (j = 0; j < 3; j++)
	{
		auprc_final_all[j] = Max(auprc_max[j][0], auprc_max[j][1]);
		auprc_final_over[j] = Max(auprc_ov[j][0], auprc_ov[j][1]);
		if (auprc_max[j][0] >= auprc_max[j][1])
		{
			strand_final_all[j] = '+';
			shift_final_all[j] = wshift[j][j_best[j][0]];
		}
		else
		{
			strand_final_all[j] = '-';
			shift_final_all[j] = wshift[j][j_best[j][1]];
		}
		if (auprc_ov[j][0] >= auprc_ov[j][1])
		{
			strand_final_over[j] = '+';
			shift_final_over[j] = wshift[j][j_ov[j][0]];
		}
		else
		{
			strand_final_over[j] = '-';
			shift_final_over[j] = wshift[j][j_ov[j][1]];
		}
	}*/
	double maxi = Max(auprc_final_over[1], auprc_final_over[2]);
	auprc_final_over1 = auprc_final_over[0] / maxi;
	if (auprc_final_over1 > 1)auprc_final_over1 = 1;
	maxi = Max(auprc_final_all[1], auprc_final_all[2]);
	auprc_final_all1 = auprc_final_all[0] / maxi;
	if (auprc_final_all1 > 1)auprc_final_all1 = 1;
	delete[] errs;
	for (k = 0; k < 2; k++)
	{
		for (j = 0; j < 2; j++)
		{
			delete[] err_inx[k][j];
		}
		delete[] err_inx[k];
	}
	delete[] err_inx;
	for (j = 0; j < 3; j++)
	{
		for (k = 0; k < 2; k++)
		{
			for (i = 0; i < n_shift[j]; i++)delete[] tp[j][k][i];
			delete[] tp[j][k];
		}
		delete[] tp[j];
	}
	delete[] tp;
	for (j = 0; j < 3; j++)delete[] fp[j];
	delete[] fp;
	for (j = 0; j < 3; j++)
	{
		for (k = 0; k < 2; k++)delete[] tp_tot[j][k];
		delete[] tp_tot[j];
	}
	delete[] tp_tot;
	for (j = 0; j < 3; j++)
	{
		for (k = 0; k < 2; k++)
		{
			delete[] auprc[j][k];
		}
		delete[] auprc[j];
	}
	delete[] auprc;
	for (k = 0; k < 2; k++)
	{
		delete[] inx_self[k];
	}
	for (j = 0; j < 3; j++)delete[] wshift[j];
	delete[] wshift;
	return 1;
}
int UnderStol(char* str, int nstol, char* ret, size_t size, char sep)
{
	memset(ret, 0, size);
	int p1, p2, len;
	if (nstol == 0)
	{
		p2 = StrNStr(str, sep, 1);
		if (p2 == -1)p2 = strlen(str);
		strncpy(ret, str, p2);
		ret[p2] = '\0';
		return 1;
	}
	else
	{
		p1 = StrNStr(str, sep, nstol);
		p2 = StrNStr(str, sep, nstol + 1);
		if (p2 == -1)
		{
			p2 = strlen(str);
		}
		if (p1 == -1 || p2 == -1) return -1;
		len = p2 - p1 - 1;
		strncpy(ret, &str[p1 + 1], len);
		ret[len] = '\0';
		return 1;
	}
}
int main(int argc, char* argv[])
{
	int i, k, mot;
	char file_fasta[ARGLEN], file_model[2][ARGLEN], type_model[2][4], file_table[2][ARGLEN];
	char file_hist[ARGLEN], file_prc[ARGLEN], file_short_all[ARGLEN], file_short_over[ARGLEN], file_sta_long[ARGLEN];
	char*** seq;// peaks
	double*** pwm;
	city sta[2];
	int model_type[2] = { -1,-1 };// 0 pwm 1 sga

	if (argc != 17)
	{
		fprintf(stderr, "Syntax error: %s 1file_fasta 2motif1_type 3motif2_type 4file_motif1_matrix 5file_motif2_matrix 6file_motif1_table 7file_motif2_table 8int max_shift_of_motif_centers", argv[0]);
		fprintf(stderr, "9double pvalue_thr 10file out_hist 11yes,0no out_hist 12file_out_prc 13int 1yes,0no out_prc 14file_out_short_over 15file_out_short_all 16file_out_sta_detailed\n");
		return -1;
	}
	strcpy(file_fasta, argv[1]);
	strcpy(type_model[0], argv[2]);//pwm or sga - type
	strcpy(type_model[1], argv[3]);//pwm or sga - type
	strcpy(file_model[0], argv[4]);//pwm or sga - matrix
	strcpy(file_model[1], argv[5]);//pwm or sga - matrix
	strcpy(file_table[0], argv[6]);//pwm or sga - thr err table
	strcpy(file_table[1], argv[7]);//pwm or sga - thr err table
	int shift = atoi(argv[8]); // shift of motifs
	double pvalue = atof(argv[9]); //threshold of expected recogntion rate
	double pvalue_lg = -log10(pvalue);
	strcpy(file_hist, argv[10]);
	int yes_out_hist = atoi(argv[11]);
	strcpy(file_prc, argv[12]);
	int yes_out_prc = atoi(argv[13]);
	strcpy(file_short_over, argv[14]);
	strcpy(file_short_all, argv[15]);
	strcpy(file_sta_long, argv[16]);

	{
		char pwm1[] = "pwm", pwm2[] = "PWM", sga1[] = "sga", sga2[] = "SGA";
		for (i = 0; i < 2; i++)
		{
			if (strcmp(type_model[i], pwm1) == 0 || strcmp(type_model[i], pwm2) == 0)
			{
				model_type[i] = 0;
			}
			if (strcmp(type_model[i], sga1) == 0 || strcmp(type_model[i], sga2) == 0)
			{
				model_type[i] = 1;
			}

		}
		for (i = 0; i < 2; i++)
		{
			if (model_type[i] == -1)
			{
				printf("Model type %d %s is not recognized\n", i + 1, type_model[i]);
				exit(1);
			}
		}

	}
	int length_fasta_max = 0, nseq_real = 0;
	seq = NULL;
	int ftp = fasta_to_plain0(file_fasta, length_fasta_max, nseq_real);
	if (ftp == -1)
	{
		fprintf(stderr, "Error: Fasta file %s error\n", file_fasta);
		return -1;
	}
	int* peak_len_real;
	peak_len_real = new int[nseq_real];
	if (peak_len_real == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }

	seq = new char** [2];
	if (seq == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
	for (k = 0; k < 2; k++)
	{
		seq[k] = new char* [nseq_real];
		if (seq[k] == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
		for (i = 0; i < nseq_real; i++)
		{
			int length_fasta_max1 = length_fasta_max + 1;
			seq[k][i] = new char[length_fasta_max1];
			if (seq[k][i] == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
			memset(seq[k][i], '\0', length_fasta_max1);
		}
	}
	pwm = new double** [2];
	if (pwm == NULL) { fprintf(stderr, "PWM Out of memory..."); return -1; }
	for (k = 0; k < 2; k++)
	{
		pwm[k] = new double* [MATLEN];
		if (pwm[k] == NULL) { fprintf(stderr, "PWM Out of memory..."); return -1; }
		for (i = 0; i < MATLEN; i++)
		{
			pwm[k][i] = new double[OLIGNUM];
			if (pwm[k][i] == NULL) { fprintf(stderr, "PWM Out of memory..."); return -1; }
		}
	}

	ftp = fasta_to_plain1(file_fasta, length_fasta_max, nseq_real, seq, peak_len_real);
	if (ftp == -1)
	{
		fprintf(stderr, "File %s error 2nd stage\n", file_fasta);
		return -1;
	}
	int olen[2];
	int nthr_dist[2];
	double min[2] = { 0,0 }, raz[2] = { 0,0 };

	double** thr_all;
	thr_all = new double* [2];
	if (thr_all == NULL) { fprintf(stderr, "Thr_all Out of memory..."); return -1; }
	double** fpr_all;
	fpr_all = new double* [2];
	if (fpr_all == NULL) { fprintf(stderr, "Thr_all Out of memory..."); return -1; }
	for (mot = 0; mot < 2; mot++)
	{
		//printf("Mot %d\n", mot);
		nthr_dist[mot] = 0;
		FILE* in_tab;
		if ((in_tab = fopen(file_table[mot], "rt")) == NULL)
		{
			printf("Input file %s can't be opened!", file_table[mot]);
			return -1;
		}
		char d[ARGLEN];
		//fgets(d, sizeof(d), in_tab);//header
		while (fgets(d, sizeof(d), in_tab) != NULL)
		{
			char c = d[0];
			char sep = '\t';
			if (c == '-' || isdigit(c))
			{
				char s[30];
				int test = UnderStol(d, 1, s, sizeof(s), sep);
				if (test == -1) { printf("Wrong format %s\n", d); exit(1); }
				nthr_dist[mot]++;
				double fprx = atof(s);
				if (fprx < pvalue_lg)break;
			}
		}
		rewind(in_tab);
		thr_all[mot] = new double[nthr_dist[mot]];
		if (thr_all[mot] == NULL) { puts("thr_all Out of memory..."); return -1; }
		fpr_all[mot] = new double[nthr_dist[mot]];
		if (fpr_all[mot] == NULL) { puts("fpr_all Out of memory..."); return -1; }
		k = 0;
		while (fgets(d, sizeof(d), in_tab) != NULL)
		{
			char c = d[0];
			if (c == '-' || isdigit(c))
			{
				char s[30];
				char sep = '\t';
				int test = UnderStol(d, 1, s, sizeof(s), sep);
				if (test == -1) { printf("Wrong format %s\n", d); exit(1); }
				thr_all[mot][k] = atof(d);
				fpr_all[mot][k] = atof(s);
				if(fpr_all[mot][k] < pvalue_lg)break;
				k++;
			}
		}
		fclose(in_tab);
	}
	for (mot = 0; mot < 2; mot++)
	{
		//printf("Mot %d\n", mot);
		if (model_type[mot] == 0)
		{
			int test = pfm_to_pwm(file_model[mot], pwm[mot]);
			if (test == -1)return -1;
			else olen[mot] = test;
			PWMScore(pwm[mot], min[mot], raz[mot], olen[mot]);
		}
		else
		{
			if (sta[mot].get_file(file_model[mot]) == -1)
			{
				printf("Site %s function not found!", file_model[mot]);
				exit(1);
			}
			olen[mot] = sta[mot].len;
		}
	}
	double auprc_final_all[3], auprc_final_over[3];
	double auprc_final_over1 = 0, auprc_final_all1 = 0;
	char strand_final_all[4], strand_final_over[4];
	for (k = 0; k < 3; k++)strand_final_all[k] = strand_final_over[k] = '+';
	strand_final_all[3] = strand_final_over[3] = '\0';
	double shift_final_all[3] = {0,0,0}, shift_final_over[3] = { 0,0,0 };
	for (k = 0; k < 3; k++)shift_final_all[k] = shift_final_over[k] = 0;
	PWM_SGA_rec_real(pwm, min, raz, sta, model_type, nthr_dist, thr_all, fpr_all, seq, olen, nseq_real, shift, length_fasta_max, file_hist, file_prc, yes_out_hist,yes_out_prc,
		auprc_final_all, auprc_final_over, auprc_final_over1, auprc_final_all1, shift_final_all, shift_final_over, strand_final_all, strand_final_over);
	FILE* out_sta_long;
	if ((out_sta_long = fopen(file_sta_long, "at")) == NULL)
	{
		printf("Output file can't be opened!\n");
		exit(1);
	}
	fprintf(out_sta_long, "%s\t%s", file_model[0], file_model[1]);
	fprintf(out_sta_long, "\tOverlap\t%f", auprc_final_over1);
	fprintf(out_sta_long, "\tAll\t%f", auprc_final_all1);
	fprintf(out_sta_long, "\tHeterotypic");
	//overap
	fprintf(out_sta_long, "\t%f", auprc_final_over[0]);
	fprintf(out_sta_long, "\t%.1f", shift_final_over[0]);
	fprintf(out_sta_long, "\t%c", strand_final_over[0]);
	//all
	fprintf(out_sta_long, "\t%f", auprc_final_all[0]);
	fprintf(out_sta_long, "\t%.1f", shift_final_all[0]);
	fprintf(out_sta_long, "\t%c", strand_final_all[0]);
	fprintf(out_sta_long, "\tHomotypic");
	//overap
	for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%f", auprc_final_over[i]);
	for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%.1f", shift_final_over[i]);
	fprintf(out_sta_long, "\t");
	for (i = 1; i < 3; i++)fprintf(out_sta_long, "%c", strand_final_over[i]);
	//all
	for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%f", auprc_final_all[i]);
	for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%.1f", shift_final_all[i]);
	fprintf(out_sta_long, "\t");
	for (i = 1; i < 3; i++)fprintf(out_sta_long, "%c", strand_final_all[i]);
	fprintf(out_sta_long, "\n");
	fclose(out_sta_long);
	FILE* out_short_over;
	if ((out_short_over = fopen(file_short_over, "at")) == NULL)
	{
		printf("Output file can't be opened!\n");
		exit(1);
	}
	fprintf(out_short_over, "\t%f", auprc_final_over1);
	fclose(out_short_over);

	FILE* out_short_all;
	if ((out_short_all = fopen(file_short_all, "at")) == NULL)
	{
		printf("Output file can't be opened!\n");
		exit(1);
	}
	fprintf(out_short_all, "\t%f", auprc_final_all1);
	fclose(out_short_all);

	for (k = 0; k < 2; k++)
	{
		delete[] thr_all[k];
	}
	delete[] thr_all;
	for (k = 0; k < 2; k++)
	{
		delete[] fpr_all[k];
	}
	delete[] fpr_all;
	delete[] peak_len_real;
	for (k = 0; k < 2; k++)
	{
		for (i = 0; i < nseq_real; i++)
		{
			delete[] seq[k][i];
		}
		delete[] seq[k];
	}
	delete[] seq;
	for (k = 0; k < 2; k++)
	{
		for (i = 0; i < MATLEN; i++)
		{
			delete[] pwm[k][i];
		}
		delete[] pwm[k];
	}
	delete[] pwm;
	return 0;
}

// Function to run the main computational logic
extern "C" int run_motali_cpp(
    const char* file_fasta,
    const char* type_model_1,
    const char* type_model_2,
    const char* file_model_1,
    const char* file_model_2,
    const char* file_table_1,
    const char* file_table_2,
    int shift,
    double pvalue,
    const char* file_hist,
    int yes_out_hist,
    const char* file_prc,
    int yes_out_prc,
    const char* file_short_over,
    const char* file_short_all,
    const char* file_sta_long
) {
    int i, k, mot;
    char file_fasta_local[ARGLEN], file_model[2][ARGLEN], type_model[2][4], file_table[2][ARGLEN];
    char file_hist_local[ARGLEN], file_prc_local[ARGLEN], file_short_all_local[ARGLEN], file_short_over_local[ARGLEN], file_sta_long_local[ARGLEN];
    char*** seq;// peaks
    double*** pwm;
    city sta[2];
    int model_type[2] = { -1,-1 };// 0 pwm 1 sga

    // Copy input parameters to local variables
    strcpy(file_fasta_local, file_fasta);
    strcpy(type_model[0], type_model_1);//pwm or sga - type
    strcpy(type_model[1], type_model_2);//pwm or sga - type
    strcpy(file_model[0], file_model_1);//pwm or sga - matrix
    strcpy(file_model[1], file_model_2);//pwm or sga - matrix
    strcpy(file_table[0], file_table_1);//pwm or sga - thr err table
    strcpy(file_table[1], file_table_2);//pwm or sga - thr err table
    strcpy(file_hist_local, file_hist);
    strcpy(file_prc_local, file_prc);
    strcpy(file_short_over_local, file_short_over);
    strcpy(file_short_all_local, file_short_all);
    strcpy(file_sta_long_local, file_sta_long);

    double pvalue_lg = -log10(pvalue);

    {
        char pwm1[] = "pwm", pwm2[] = "PWM", sga1[] = "sga", sga2[] = "SGA";
        for (i = 0; i < 2; i++)
        {
            if (strcmp(type_model[i], pwm1) == 0 || strcmp(type_model[i], pwm2) == 0)
            {
                model_type[i] = 0;
            }
            if (strcmp(type_model[i], sga1) == 0 || strcmp(type_model[i], sga2) == 0)
            {
                model_type[i] = 1;
            }

        }
        for (i = 0; i < 2; i++)
        {
            if (model_type[i] == -1)
            {
                printf("Model type %d %s is not recognized\n", i + 1, type_model[i]);
                return -1; // Changed from exit(1) to return -1
            }
        }

    }
    int length_fasta_max = 0, nseq_real = 0;
    seq = NULL;
    int ftp = fasta_to_plain0(file_fasta_local, length_fasta_max, nseq_real);
    if (ftp == -1)
    {
        fprintf(stderr, "Error: Fasta file %s error\n", file_fasta_local);
        return -1;
    }
    int* peak_len_real;
    peak_len_real = new int[nseq_real];
    if (peak_len_real == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }

    seq = new char** [2];
    if (seq == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
    for (k = 0; k < 2; k++)
    {
        seq[k] = new char* [nseq_real];
        if (seq[k] == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
        for (i = 0; i < nseq_real; i++)
        {
            int length_fasta_max1 = length_fasta_max + 1;
            seq[k][i] = new char[length_fasta_max1];
            if (seq[k][i] == NULL) { fprintf(stderr, "Error: Not of memory..."); return -1; }
            memset(seq[k][i], '\0', length_fasta_max1);
        }
    }
    pwm = new double** [2];
    if (pwm == NULL) { fprintf(stderr, "PWM Out of memory..."); return -1; }
    for (k = 0; k < 2; k++)
    {
        pwm[k] = new double* [MATLEN];
        if (pwm[k] == NULL) { fprintf(stderr, "PWM Out of memory..."); return -1; }
        for (i = 0; i < MATLEN; i++)
        {
            pwm[k][i] = new double[OLIGNUM];
            if (pwm[k][i] == NULL) { fprintf(stderr, "PWM Out of memory..."); return -1; }
        }
    }

    ftp = fasta_to_plain1(file_fasta_local, length_fasta_max, nseq_real, seq, peak_len_real);
    if (ftp == -1)
    {
        fprintf(stderr, "File %s error 2nd stage\n", file_fasta_local);
        return -1;
    }
    int olen[2];
    int nthr_dist[2];
    double min[2] = { 0,0 }, raz[2] = { 0,0 };

    double** thr_all;
    thr_all = new double* [2];
    if (thr_all == NULL) { fprintf(stderr, "Thr_all Out of memory..."); return -1; }
    double** fpr_all;
    fpr_all = new double* [2];
    if (fpr_all == NULL) { fprintf(stderr, "Thr_all Out of memory..."); return -1; }
    for (mot = 0; mot < 2; mot++)
    {
        //printf("Mot %d\n", mot);
        nthr_dist[mot] = 0;
        FILE* in_tab;
        if ((in_tab = fopen(file_table[mot], "rt")) == NULL)
        {
            printf("Input file %s can't be opened!", file_table[mot]);
            return -1;
        }
        char d[ARGLEN];
        //fgets(d, sizeof(d), in_tab);//header
        while (fgets(d, sizeof(d), in_tab) != NULL)
        {
            char c = d[0];
            char sep = '\t';
            if (c == '-' || isdigit(c))
            {
                char s[30];
                int test = UnderStol(d, 1, s, sizeof(s), sep);
                if (test == -1) { printf("Wrong format %s\n", d); return -1; } // Changed from exit(1) to return -1
                nthr_dist[mot]++;
                double fprx = atof(s);
                if (fprx < pvalue_lg)break;
            }
        }
        rewind(in_tab);
        thr_all[mot] = new double[nthr_dist[mot]];
        if (thr_all[mot] == NULL) { puts("thr_all Out of memory..."); return -1; }
        fpr_all[mot] = new double[nthr_dist[mot]];
        if (fpr_all[mot] == NULL) { puts("fpr_all Out of memory..."); return -1; }
        k = 0;
        while (fgets(d, sizeof(d), in_tab) != NULL)
        {
            char c = d[0];
            if (c == '-' || isdigit(c))
            {
                char s[30];
                char sep = '\t';
                int test = UnderStol(d, 1, s, sizeof(s), sep);
                if (test == -1) { printf("Wrong format %s\n", d); return -1; } // Changed from exit(1) to return -1
                thr_all[mot][k] = atof(d);
                fpr_all[mot][k] = atof(s);
                if(fpr_all[mot][k] < pvalue_lg)break;
                k++;
            }
        }
        fclose(in_tab);
    }
    for (mot = 0; mot < 2; mot++)
    {
        //printf("Mot %d\n", mot);
        if (model_type[mot] == 0)
        {
            int test = pfm_to_pwm(file_model[mot], pwm[mot]);
            if (test == -1)return -1;
            else olen[mot] = test;
            PWMScore(pwm[mot], min[mot], raz[mot], olen[mot]);
        }
        else
        {
            if (sta[mot].get_file(file_model[mot]) == -1)
            {
                printf("Site %s function not found!", file_model[mot]);
                return -1; // Changed from exit(1) to return -1
            }
            olen[mot] = sta[mot].len;
        }
    }
    double auprc_final_all[3], auprc_final_over[3];
    double auprc_final_over1 = 0, auprc_final_all1 = 0;
    char strand_final_all[4], strand_final_over[4];
    for (k = 0; k < 3; k++)strand_final_all[k] = strand_final_over[k] = '+';
    strand_final_all[3] = strand_final_over[3] = '\0';
    double shift_final_all[3] = {0,0,0}, shift_final_over[3] = { 0,0,0 };
    for (k = 0; k < 3; k++)shift_final_all[k] = shift_final_over[k] = 0;
    PWM_SGA_rec_real(pwm, min, raz, sta, model_type, nthr_dist, thr_all, fpr_all, seq, olen, nseq_real, shift, length_fasta_max, file_hist_local, file_prc_local, yes_out_hist, yes_out_prc,
        auprc_final_all, auprc_final_over, auprc_final_over1, auprc_final_all1, shift_final_all, shift_final_over, strand_final_all, strand_final_over);
    
    // Output results to files
    FILE* out_sta_long;
    if ((out_sta_long = fopen(file_sta_long_local, "at")) == NULL)
    {
        printf("Output file can't be opened!\n");
        return -1; // Changed from exit(1) to return -1
    }
    fprintf(out_sta_long, "%s\t%s", file_model[0], file_model[1]);
    fprintf(out_sta_long, "\tOverlap\t%f", auprc_final_over1);
    fprintf(out_sta_long, "\tAll\t%f", auprc_final_all1);
    fprintf(out_sta_long, "\tHeterotypic");
    //overap
    fprintf(out_sta_long, "\t%f", auprc_final_over[0]);
    fprintf(out_sta_long, "\t%.1f", shift_final_over[0]);
    fprintf(out_sta_long, "\t%c", strand_final_over[0]);
    //all
    fprintf(out_sta_long, "\t%f", auprc_final_all[0]);
    fprintf(out_sta_long, "\t%.1f", shift_final_all[0]);
    fprintf(out_sta_long, "\t%c", strand_final_all[0]);
    fprintf(out_sta_long, "\tHomotypic");
    //overap
    for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%f", auprc_final_over[i]);
    for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%.1f", shift_final_over[i]);
    fprintf(out_sta_long, "\t");
    for (i = 1; i < 3; i++)fprintf(out_sta_long, "%c", strand_final_over[i]);
    //all
    for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%f", auprc_final_all[i]);
    for (i = 1; i < 3; i++)fprintf(out_sta_long, "\t%.1f", shift_final_all[i]);
    fprintf(out_sta_long, "\t");
    for (i = 1; i < 3; i++)fprintf(out_sta_long, "%c", strand_final_all[i]);
    fprintf(out_sta_long, "\n");
    fclose(out_sta_long);
    
    FILE* out_short_over;
    if ((out_short_over = fopen(file_short_over_local, "at")) == NULL)
    {
        printf("Output file can't be opened!\n");
        return -1; // Changed from exit(1) to return -1
    }
    fprintf(out_short_over, "\t%f", auprc_final_over1);
    fclose(out_short_over);

    FILE* out_short_all;
    if ((out_short_all = fopen(file_short_all_local, "at")) == NULL)
    {
        printf("Output file can't be opened!\n");
        return -1; // Changed from exit(1) to return -1
    }
    fprintf(out_short_all, "\t%f", auprc_final_all1);
    fclose(out_short_all);

    // Cleanup allocated memory
    for (k = 0; k < 2; k++)
    {
        delete[] thr_all[k];
    }
    delete[] thr_all;
    for (k = 0; k < 2; k++)
    {
        delete[] fpr_all[k];
    }
    delete[] fpr_all;
    delete[] peak_len_real;
    for (k = 0; k < 2; k++)
    {
        for (i = 0; i < nseq_real; i++)
        {
            delete[] seq[k][i];
        }
        delete[] seq[k];
    }
    delete[] seq;
    for (k = 0; k < 2; k++)
    {
        for (i = 0; i < MATLEN; i++)
        {
            delete[] pwm[k][i];
        }
        delete[] pwm[k];
    }
    delete[] pwm;
    
    return 0;
}
