// BBB -- Beerenberg backup builder
//Picks up each month disk deposits ledger accounts into current directory
// Also saves INVOICES.DBF,JOURNAL.DBF and PURCHASE.DBF

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <alloc.h>
#include <conio.h>
#include <iostream.h>
#include <dir.h>

int main()
{
  char *fmon,*mon,*nyear;
  char *fn=(char *) malloc(12);
  char *suffix=(char *) malloc(5);
  char *command=(char *) malloc(140);
  char *path="c:\\dbase\\archive";
  char oldpath[MAXPATH];
  int key,drive;
  int count,year;

  getcwd(oldpath,MAXPATH);
//  window(1,10,80,17);
  clrscr();
//  textattr(BLACK + YELLOW);
//  do
//  {
     printf("\n\n\n");
     printf("B E E R E N B E R G  A R C H I V E  A D D E R\n");
     printf("=============================================\n\n\n");
     printf("This program will add an extra backup disk into the archive.\n");
     printf("Please enter the disk drive you want to use.\n");
     printf("(A) for A: Large 5.25 inch floppy disks\n");
     printf("(B) for B: Small 3.5 inch disks\n\n");
     do drive=getch(); while (!(drive =='A' || drive =='a' || drive =='B' || drive =='b'));
//
     if (drive=='A' || drive=='a')
	printf("Drive A: selected....Large Floppy disks\n\n");
     else
	printf("Drive B: selected....Small 3.5 inch disks\n\n");
//
     printf("Please enter month of backup disk you wish to add (1-12, 0 to exit):");
     do {
	cin.width(2);
	cin >> count;
	}
     while (!(count >=0 && count <= 12));
     if (count != 0)
     {
	printf("Please enter year of backup disk you wish to add. e.g.'94':");
	do {
	   cin.width(2);
	   cin >> year;
	   itoa(year,nyear,10);
	   }
	while (!(year >= 0 && year <= 99));
	printf("\n\n");
	switch (count)
	{
	   case 1:{ mon="JAN";
		 fmon="January";
		 break;  }
	   case 2:{ mon="FEB";
		 fmon="February";
		 break;  }
	   case 3:{ mon="MAR";
		 fmon="March";
		 break;  }
	   case 4:{ mon="APR";
		 fmon="April";
		 break;  }
	   case 5:{ mon="MAY";
		 fmon="May";
		 break;  }
	   case 6:{ mon="JUN";
		 fmon="June";
		 break;  }
	   case 7:{ mon="JUL";
		 fmon="July";
		 break;  }
	   case 8:{ mon="AUG";
		 fmon="August";
		 break;  }
	   case 9:{ mon="SEP";
		 fmon="September";
		 break;  }
	   case 10:{ mon="OCT";
		  fmon="October";
		  break;  }
	   case 11:{ mon="NOV";
		  fmon="November";
		  break;  }
	   case 12:{ mon="DEC";
		  fmon="December"; }
	   }
	   printf("Please insert backup disk for %s '%d\n",fmon,year);
	   printf("Hit enter to continue or X to exit.\n");
	   do key=getch(); while (!(key=='X' || key=='x' || key==13));
	   if (!(key=='X' || key=='x'))
	   {
	      chdir(path);
	      strcpy(suffix,"     ");
	      strcpy(suffix,mon);
	      strcat(suffix,nyear);
	      strcpy(fn,"     ");
	      strcpy(fn,"BS3");
	      strcat(fn,suffix);
	      strcat(fn,".zip");
	//      printf("fn=%s\n",fn);
	      printf("Looking for file : %s\n\n",fn);
	//      printf("%s\n",suffix);
	      if (drive=='A' || drive=='a')
	      {
		 system("copy a:open_bal.dbf");
		 system("copy a:account.dbf");
		 system("copy a:profit.dbf");
		 system("copy a:invoices.dbf");
		 system("copy a:purchase.dbf");
		 system("copy a:journal.dbf");
		 system("copy a:pay_in.dbf");
		 system("copy a:pay_out.dbf");
	      }
	//      printf("fn=%s\n",fn);
	//      printf("%s\n",suffix);
//
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"pkunzip");
	//      printf("fn=%s\n",fn);
	      if (drive=='A' || drive=='a')
		 strcat(command," a:");
	      else
		 strcat(command," b:");
	 //     printf("fn=%s\n",fn);
//
	      strcat(command,fn);
	      strcat(command," open_bal.dbf account.dbf profit.dbf invoices.dbf purchase.dbf journal.dbf pay_in.dbf pay_out.dbf");
	 //     printf("%s\n",command);
	      system(command);
//
// Rename files by month and date
// Open_bal
	      strcpy(command,"                                                                                                                                             ");
       //	      printf("%s\n",suffix);
	      strcpy(command,"rename ");
	      strcat(command,"open_bal.dbf ");
	      strcat(command,"OB");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del open_bal.dbf");
//  Account
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"rename ");
	      strcat(command,"account.dbf ");
	      strcat(command,"AC");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del account.dbf");
// Profit
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"rename ");
	      strcat(command,"profit.dbf ");
	      strcat(command,"PR");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del profit.dbf");
//
//  Invoices
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"rename ");
	      strcat(command,"invoices.dbf ");
	      strcat(command,"IN");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del invoices.dbf");
// Purchases
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"rename ");
	      strcat(command,"purchase.dbf ");
	      strcat(command,"PU");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del purchase.dbf");
//   Journal
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"rename ");
	      strcat(command,"journal.dbf ");
	      strcat(command,"JO");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del journal.dbf");
//   Payments in
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"rename ");
	      strcat(command,"pay_in.dbf ");
	      strcat(command,"PI");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del pay_in.dbf");
//   Payments out
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(command,"rename ");
	      strcat(command,"pay_out.dbf ");
	      strcat(command,"PO");
	      strcat(command,suffix);
	      strcat(command,".dbf");
	      printf("%s\n",command);
	      system(command);
	      system("del pay_out.dbf");
//
	      strcpy(command,"                                                                                                                                             ");
	      strcpy(fn,"             ");
	      strcpy(suffix,"     ");
	 }
      }
   if (!(key=='X' || key=='x' || count==0))
   {
      printf("Year to date data extracted.....Hit any key\n");
      key=getch();
   }
   else
   {
      printf("\n\nProgram aborted....No archives added.\n");
      printf("Hit any key to exit\n");
      key=getch();
   }
   chdir(oldpath);
   //
   free(command);
   free(fn);
   free(suffix);
return 0;
}
