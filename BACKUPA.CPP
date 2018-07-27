// BACKUP3  -- compresses using pkzip

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <conio.h>
#include <dir.h>
#include <iostream.h>

int main()
{
  char month[3],year[2],fn[8],fm[9];
  char command[80],command2[80],command3[80];  //,command4[80];
  char *fnp=fn,*mon=month,*yp=year,*fmon=fm;
  int mn,yn,pos,count,nyear;
  int key,key2;

  clrscr();
  textattr(BLACK + YELLOW);
  cprintf("\n\nBackup Utility Ver 3.1 - By Anthony Paech\r\n\n");
  cprintf("Please enter month of backup disk you wish to add (1-12, 0 to exit):");
  do cin >> count; while (!(count >=0 && count <= 12));
  cprintf("\r\n");
  if (count != 0)
  {
     cprintf("Please enter year of backup disk you wish to add. e.g.'94':");
     do cin >> nyear; while (!(nyear >= 0 && nyear <= 99));
     cprintf("\r\n");
     itoa(nyear,yp,10);
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
     cprintf("Please use backup disk for %s '%s\r\n",fmon,yp);
     //printf("Hit enter to continue or X to exit.\n");
     //do key=getch(); while (!(key=='X' || key=='x' || key==13));
     strcpy(fnp,"BS3");
     strcat(fnp,mon);
     strcat(fnp,yp);
     strcat(fnp,".zip");
    // printf("Backup file name  : %s\n",fnp);
    // printf("Are you still in accounting month %s (%d)?\n",mon,mn);
     cprintf("Are you sure you want to backup under %s (Y/N)?",fnp);
     do {
	do {
	   key=getche();
	   cprintf("\b");
	} while (!(key=='Y' || key=='N' || key=='y' || key=='n'));
	key2=getch();
     } while (key2!=13);
     cprintf("\r\n");
     if (key=='Y' || key=='y')
     {
// compress *.dbf *.mdx files into file called fnp in current directory
	cprintf("Backing up files to hard disk....Please wait.\r\n");
	strcpy(command,"pkzip ");
	strcat(command,fnp);
	strcat(command," c:\\dbase\\bsfiles3\\*.dbf c:\\dbase\\bsfiles3\\*.mdx");
//	cprintf("%s\r\n",command);
	system(command);
// Copy backup file from current directory to c:\backup.
	strcpy(command2,"copy ");
	strcat(command2,fnp);
	strcat(command2," c:\\backup");
//	system("pause");
//	cprintf("%s\r\n",command2);
	system(command2);
// Copy backup file to drive a: (first backup disk)
	cprintf("\n\nPlease insert first backup disk into drive A:\r\n");
	strcpy(command3,"copy ");
	strcat(command3,fnp);
	strcat(command3," a:");
	system("pause");
	cprintf("%s\r\n",command3);
	system(command3);
 // Copy backup file to second backup disk
	cprintf("\n\nPlease insert second backup disk into drive A:\r\n");
	system("pause");
	system("del a:bs3*.zip");
	cprintf("%s\r\n",command3);
	system(command3);
// housekeeping del temp file
//	strcpy(command4,"del ");
//	strcat(command4,"bs3*.zip");
//	cprintf("%s\r\n",command4);
//	system(command4);
	cprintf("\nBackup complete! ...Hit any key\r\n");
	getch();
	cprintf("\r\n\n\n");
     }
     else
     {
	cprintf("\r\n\n");
	cprintf("Backup to hard drive aborted...no backup made for today.\r\n");
	cprintf("Hit any key....\r\n");
	getch();
	cprintf("\r\n\n\n");
     }
  }
return 0;
}

