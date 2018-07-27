// USE  -- picks up compressed (using pkzip) file from current directory

#include <string.h>
#include <dos.h>
#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>

int main()
{
  char month[3],year[2],fn[8];
  char command[80];
  char *fnp=fn,*mon=month,*yp=year;
  int mn,yn,ans;

  do {
     printf("Enter month number of backup disk eg.1 to 12. (0 to Exit):");
     do {
	cin >> mn ;
	switch (mn)
	{
	   case 1:mon="JAN";break;
	   case 2:mon="FEB";break;
	   case 3:mon="MAR";break;
	   case 4:mon="APR";break;
	   case 5:mon="MAY";break;
	   case 6:mon="JUN";break;
	   case 7:mon="JUL";break;
	   case 8:mon="AUG";break;
	   case 9:mon="SEP";break;
	   case 10:mon="OCT";break;
	   case 11:mon="NOV";break;
	   case 12:mon="DEC";
	}
     } while ((mn < 0) || (mn > 12));
     if (mn!=0)
     {
	printf("        Month:%s\n",mon);
	printf("Enter year of backup disk. eg.94:");
	do cin >> yn ; while ((yn < 0) || (yn > 99));
	yp=itoa(yn,yp,10);
	printf("\n");
	strcpy(fnp,"BS3");
	strcat(fnp,mon);
	strcat(fnp,yp);
	strcat(fnp,".zip");
	printf("Looking for filename :%s\n",fnp);
	printf("Is this correct? (Y/N)");
	do ans=getch(); while (!(ans=='Y' || ans=='y' || ans=='N' || ans=='n'));
	printf("\n");
     }
  } while (!(ans=='Y' || ans=='y' || mn==0));
  if (mn!=0)
  {
     strcpy(command,"pkunzip ");
     strcat(command,fnp);
     strcat(command," c:\\dbase\\oldmonth");
     printf("%s\n",command);
     system(command);
     printf("\nOld files are now ready to use. Please remove backup disk\n");
  }
  else
     printf("\nUse backup disk program aborted. No files retreived.\n");
  system("pause");
  return 0;
}


