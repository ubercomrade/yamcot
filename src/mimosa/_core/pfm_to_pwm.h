double pfm[MATLEN][OLIGNUM];

int NthColumn(char *din, char *dout, int nstol)
{
	int p1, p2;
	if(nstol==0)p1=-1;
	else 
	{
		p1=StrNStr(din,'\t',nstol);
		if(p1==-1)return -1;
	}
	p2=StrNStr(din,'\t',nstol+1);
	if(p2==-1)p2=strlen(din);
	int len=p2-p1-1;
	strncpy(dout,&din[p1+1],len);	
	if(len==0)return -1;
	dout[len]='\0';
	return 1;
}
int pfm_to_pwm(char *file_pfm, double **mat)
{	
	char d[500],head[500], s[500];	
    int i, j, len=0; 
	double nseq;//=atoi(argv[3]);
	int shift_col;//=atoi(argv[4]);
	
	FILE *in_pfm;
	if((in_pfm=fopen(file_pfm,"rt"))==NULL)
	{
		printf("Input file %s can't be opened!\n",file_pfm);
		return -1;
	}
	int alfabet=0;
	fgets(head,sizeof(head),in_pfm);			 
	fgets(head,sizeof(head),in_pfm);	
	//homer, cis-bp
	nseq=1000000000;
	//shift_col=0;		
	{
		DelChar(head,'\n');
		DelChar(head,'\r');
		int headlen= strlen(head);
		if(head[headlen-1]=='\t')
		{
			head[headlen-1]='\0';
			headlen--;
		}
		int counttab=0;
		for(i=0;i<headlen;i++)
		{
			if(head[i]=='\t')counttab++;
		}
		if(counttab==4 || counttab==16)
		{
			shift_col=1;
			alfabet=counttab;
		}
		else 
		{
			if(counttab==3 || counttab==15)
			{
				shift_col=0;
				alfabet=counttab+1;
			}
			else
			{
				printf("Reading errorin Cisbp/Homer matrix file %s !\n",file_pfm);
				fclose(in_pfm);
				return -1;
			}
		}
	}
	rewind(in_pfm);
	
	int olen=0;
	fgets(head,sizeof(head),in_pfm);		
	for(i=0;i<MATLEN;i++)
	{
		if(fgets(d,sizeof(d),in_pfm)!=NULL)
		{
			char c=d[0];
			if(isdigit(c) || (strchr("-ATGC",c)!=0))olen++;
		}
		else break;
	}
	rewind(in_pfm);
	fgets(head,sizeof(head),in_pfm);		
	for(i=0;i<MATLEN;i++)
	{
		if(fgets(d,sizeof(d),in_pfm)!=NULL)
		{
	//		DelChar(d,'\n');
	//		DelChar(d,'\r');
			char c=d[0];
			if(isdigit(c) || (strchr("-ATGC",c)!=0))
			{				
				//	 printf("%s",d[i]);						
				for(j=0;j<alfabet;j++)
				{			
					if(NthColumn(d,s,j+shift_col)==-1)
					{
						break;			
					}				
					//double score=atof(s);
					//pfm[i][j]=nseq*score;
					pfm[i][j]=atof(s);
				}					
			}
			else break;
		}
		else break;
	}		
	//logodds score
	double pse=0.25;
	double pse4=1;
	double vych=log10(pse);		
	int olen1=olen-1;
	for(i=0;i<olen;i++)
	{			
		double sum=0;
		for(j=0;j<alfabet;j++)sum+=nseq*pfm[i][j];
		int alfabet1=alfabet-1;
		for(j=0;j<alfabet;j++)
		{
			double count=nseq*pfm[i][j];
			double ves=(count+pse)/(sum+pse4);
			mat[i][j]=log10(ves)-vych;
		}			
	}
	fclose(in_pfm);
	return olen;
}
