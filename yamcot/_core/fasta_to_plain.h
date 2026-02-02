char TransStr(char x)
{
  int c=int(x);
  if(c<97) x=char(c+32);	
   return x;
}
int fasta_to_plain_genome(char* file_in_fasta, int motif_len_min, int& all_pos, int &nseq, int &len)
{
	char l[SEQLEN], d[SEQLEN], head[400];
	int fl = 0;
	FILE* in;

	if ((in = fopen(file_in_fasta, "rt")) == NULL)
	{
		printf("Input file %s can't be opened\n", file_in_fasta);
		return -1;
	}
	char symbol = fgetc(in);
	rewind(in);
	all_pos = nseq = len = 0;
	while (nseq >= 0)
	{
		if (fgets(l, sizeof(l), in) == NULL) fl = -1;
		if (*l == '\n' && fl != -1)continue;
		if (((*l == symbol) || (fl == -1)) && (fl != 0))
		{
			int lenx = (int)strlen(d);
			if (lenx > len)len = lenx;
			all_pos += (lenx-motif_len_min);
			nseq++;
			//	int check = CheckStr(file, d, n, 1, outlog);
			//if (check != -1)			
			if (fl == -1)
			{
				fclose(in);				
				break;
			}			
		}
		if (*l == symbol)
		{
			memset(head, 0, sizeof(head));
			DelChar(l, '\n');
			strcpy(head, l);
			fl = 0; continue;
		}
		if (fl == 0)
		{
			memset(d, 0, sizeof(d));
			DelChar(l, '\n');
			strcpy(d, l);
			fl = 1; continue;
		}
		if (strlen(d) + strlen(l) > sizeof(d))
		{
			printf("Size is large...");
			printf("l:%s\nstrlen(l):%zu\n", l, strlen(l));
			printf("d:%s\nstrlen(d):%zu\n", d, strlen(d));
			exit(1);
		}
		DelChar(l, '\n');
		strcat(d, l);
	}
	return 1;
}
int fasta_to_plain0(char *file_in_fasta, int &length_fasta_max, int &nseq_fasta)
{
	char head[200];		
	FILE *in;

	if((in=fopen(file_in_fasta,"rt"))==NULL)
	{
		printf("Input file %s can't be opened\n",file_in_fasta);
		return -1;
	}
	char c, symbol = '>';	
	nseq_fasta=0;
	int len=0;
	//double sum_len=0;
	int fl=1;
	length_fasta_max=0;
	do  
	{
		c=getc(in);
		if(c==EOF)fl=-1;
		if(c==symbol || fl==-1)
		{
			if(nseq_fasta>0)
			{
				if(len>length_fasta_max)length_fasta_max=len;
				if(len>SEQLEN)
				{
					printf("Sequence N %d too long... %d nt\n",nseq_fasta,len);
					return -1;
				}
			}			
			if(fl!=-1)
			{
				nseq_fasta++;				
				len=0;
			}
			if(fl==1)
			{
				fgets(head,sizeof(head),in);			
				continue;
			}
		}
		if(c=='\n')continue;	
		if(c=='\t')continue;	
		if(c==' ')continue;	
		if(c=='\r')continue;	
		len++;
	}
	while(fl==1);
	fclose(in);
	return 1;
}
int fasta_to_plain1(char *file_in_fasta, int length_fasta_max, int nseq_fasta, char ***seq, int *peak_len)
{
	int fl=1, n=0, len=0;
	char head[200];		
	char c, symbol = '>';
	char alfavit4[]="acgtnACGTN";
	FILE *in;
	if((in=fopen(file_in_fasta,"rt"))==NULL)
	{
		printf("Input file %s can't be opened\n",file_in_fasta);
		return -1;
	}	
	do  
	{
		c=getc(in);
		if(c==EOF)
		{
			fl=-1;
		}
		if(c==symbol || fl==-1)
		{		
			if(len>0)
			{	
				peak_len[n]=len;
				seq[0][n][len]='\0';
				strncpy(seq[1][n],seq[0][n],len);
				seq[1][n][len]='\0';
				ComplStr(seq[1][n]);								
				{
					if(fl!=-1)
					{
						n++;						
						len=0;
					}
				}
			}
			else
			{
				if(n>0 && fl!=-1)
				{
					printf("Peak length error! peak %d\n",n+1);
					return -1;
				}
			}
			if(fl==-1)break;
			if(fl==1)
			{
				fgets(head,sizeof(head),in);			
				continue;
			}
		}
		if(c=='\n')continue;	
		if(c=='\t')continue;	
		if(c==' ')continue;	
		if(c=='\r')continue;	
		if(strchr(alfavit4,c)!=NULL)
		{
			c=TransStr(c);
			seq[0][n][len++]=c;
		}
		else seq[0][n][len++]='n';
	}
	while(fl==1);
	fclose(in);	
	return 1;
}