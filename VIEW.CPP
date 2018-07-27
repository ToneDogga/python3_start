// VIEW.CPP  -- Allows complete scrolling of an ASCII file
// improved 22/05/95  By Anthony Paech

#include <conio.h>
#include <stdio.h>
#include <malloc.h>

void fillwindow(unsigned long *);
void display_line(void);
void display_col(int,unsigned long *);
void scrollleft(void);
void scrollright(void);
unsigned long *movehome(unsigned long *,unsigned long *);
unsigned long *moveend(unsigned long *,unsigned long *,unsigned long *,unsigned long);

FILE *in;  //file pointer

const int RF = 24;   //Window finish row
const int RS = 3;    // Window start row
const int CF = 80;   // Window finish column
const int CS = 1;    // Window start column
const int WCOLOUR = 14;  // Window text colour
const int BCOLOUR = 78;   // Border text colour
const int MAXWIDTH = 170;  // Maximum column display width
const int PAGE = 21;  // page size

int col=CS;    //Left hand column position


int main(int argc, char *argv[])
{
   int k,jump,key=0;
   char c;
   unsigned long temppos,lineno,lines;  // total number of lines
   unsigned long *arrayptr,*temp;

   if (argc!=2)
   {
      cprintf("Usage: VIEW file.xxx\n");
      return(-1);
   }
   if ((in=fopen(argv[1],"rt"))==NULL)
   {
      cprintf("Cannot open file: %s for VIEWing\n",argv[1]);
      return(-1);
   }
   _setcursortype(_NOCURSOR);
   directvideo=1;    // speeds up display on 100% IBM compatible monitors
   textmode(3);          // initialise window
   textattr(BCOLOUR);
   clrscr();
   gotoxy(1,1);
   cprintf(" VIEWing file:%s\r\n",argv[1]);
   gotoxy(1,2);
   cprintf(" ------------------------------------------------------------------------------");
   gotoxy(1,25);
   cprintf(" -----Use (%c %c %c %c) ,Page Up ,Page down, Home and End to move. Esc to exit.----",24,25,26,27); //arrow characters
   textattr(WCOLOUR);  // backround black,char light green

  // Main input loop

   fseek(in,0,SEEK_SET);       // reset to BOF
   lines=1;   // first line has no CR
   do        // count lines first
   {
      c=fgetc(in);
      if (c==10)    //end of line
	 lines++;
   }  while (c!=EOF);
	// allocate memory to string of no of lines.
   if ((arrayptr=(unsigned long *) malloc(lines))==NULL)
   {
      cprintf("Not enough memory! lines(%ld) * sizeof(%d) = %d\n",lines,sizeof(unsigned long),lines*sizeof(unsigned long));
      return(-1);
   }
   unsigned long *start=arrayptr;   // save original position

   window(CS,1,CF,25);
   textattr(BCOLOUR);
   gotoxy(61,1);
   cprintf("Lines:      /%06ld",lines);
   fseek(in,0,SEEK_SET);
   lineno=0;
   *arrayptr++=0;    // First line file starting position
   do     // fill line_end structure string with file positions at end of lines
   {
      c=fgetc(in);
      if ((c==10) || (c==EOF))
      {
	 *arrayptr++=ftell(in);
	 gotoxy(67,1);
	 cprintf("%06ld",lineno++);
      }
   }  while (c!=EOF);
//
   unsigned long *finish=arrayptr;    // get final position
//
   gotoxy(50,1);
   cprintf("LH Col:%03d",col);
   gotoxy(67,1);
   cprintf("000001/%06ld",lines);   //-RS+1);
   textattr(WCOLOUR);
//
   gotoxy(1,2);
   temp=start;
   fillwindow(start);   // fill window initially
   start=arrayptr=temp;
// returns the file position is on the c=10 at the RF-RS+1 line
   lineno=1;      // position is first line
   gotoxy(CS,RS);
   if (lines < (RF-RS))
      lines=(RF-RS);
//   position pointer at line at end of window
   do
   {
	switch (key)
	{         // Arrows
	   case 72:if (lineno > 1)   // scroll down
		   {
		      movetext(CS,RS,CF,RF-1,CS,RS+1);
		      gotoxy(CS,RS);
		      temp=arrayptr;
		      arrayptr--;
		      fseek(in,*arrayptr,SEEK_SET);   // wind back file pos
		      display_line();        // display from file pos
		      arrayptr=--temp;
		      fseek(in,*arrayptr,SEEK_SET);
		      lineno--;
		      textattr(BCOLOUR);
		      gotoxy(67,1);
		      cprintf("%06ld",lineno);
		      textattr(WCOLOUR);
		   }
		   break;
	   case 80:if (lineno < lines-(RF-RS)+1)  // scroll up
		      {
			 movetext(CS,RS+1,CF,RF,CS,RS);
			 gotoxy(CS,RF);
			 temp=arrayptr;
			 arrayptr=arrayptr+(RF-RS)+1;
			 fseek(in,*arrayptr,SEEK_SET);   // wind back file pos
			 display_line();        // display from file pos
			 arrayptr=++temp;
			 fseek(in,*arrayptr,SEEK_SET);
			 lineno++;
			 textattr(BCOLOUR);
			 gotoxy(67,1);
			 cprintf("%06ld",lineno);
			 textattr(WCOLOUR);
		      }
		   break;
	   case 75:if (col > 1)
		   {
		      scrollright();
		      temp=arrayptr;
		      display_col(CS,arrayptr);  // go back to top of window
		      arrayptr=temp;         // reset file position
		      fseek(in,*arrayptr,SEEK_SET);
		   }
		   break;
	   case 77:if (col < MAXWIDTH)
		   {
		      scrollleft();
		      temp=arrayptr;        // save current file position
		      display_col(CF,arrayptr);    // wind back line position pointer to top of window
		      arrayptr=temp;         // reset saved line position pointer
		      fseek(in,*arrayptr,SEEK_SET);   // reset file position
		   }
		   break;
	   case 71:arrayptr=movehome(start,&lineno);    // home
		   break;
	   case 79:arrayptr=moveend(finish,start,&lineno,lines);  //end
		   break;
	   case 73:if (lineno > PAGE+1)
		   {
		      arrayptr=arrayptr-PAGE-1;
		      fillwindow(arrayptr);  // pageup
		      lineno=lineno-PAGE;
		      textattr(BCOLOUR);
		      gotoxy(67,1);
		      cprintf("%06ld",lineno);
		      textattr(WCOLOUR);
		   }
		   else
		      arrayptr=movehome(start,&lineno);      // goto home instead
		   break;
	   case 81:if ((lineno < lines-(RF-RS)-PAGE+1) && (lines > (RF-RS)+PAGE))
		   {
			 arrayptr+=PAGE+1;
			 fillwindow(arrayptr);  // pagedown
			 lineno=lineno+PAGE;
			 textattr(BCOLOUR);
			 gotoxy(67,1);
			 cprintf("%06ld",lineno);
			 textattr(WCOLOUR);
		   }
		   else
		      arrayptr=moveend(finish,start,&lineno,lines);    //goto end instead
	}
     key = (int) getch();
   }  while (key!=27); // escape
//
   fclose(in);
   textmode(-1);
   clrscr();
   directvideo=0;
   _setcursortype(_NORMALCURSOR);
   free(start);  // unmalloc line diff array (arrayptr)
   return(0);
}



void fillwindow(unsigned long *start_pos)
{
   char c;  //one character at a time
   int colcount,eol,pos=CS;  //chars per line
   unsigned long linecount=RS;
   unsigned long *linepos;

   linepos=start_pos;
   gotoxy(CS,RS);
   fseek(in,*start_pos,SEEK_SET);   //BOF
   while (linecount++ <= RF)
   {
      eol=0;
      for (colcount=1;((colcount < col) && (!eol));colcount++)
      {
	 c=fgetc(in);
	 if ((c==10) || (c==EOF))
	    eol=1;
      }
      if (!eol)
      {
	 do
	 {
	    c=fgetc(in);
	    if (pos++ <= CF)
	       if ((c==10) || (c==EOF))
		  if (c==10)
		     clreol();
		  else
		  {      // if eof is mid window display blanks to end of window
		     while (linecount++ <= RF)
		     {
			clreol();
			gotoxy(CS,linecount);
		     }
		     gotoxy(CS,RF);   // sometimes eof is on end of line
		     clreol();        // or at start of line by itself
		  }
	       else
		  putch(c);
	 }  while ((c!=10) && (c!=EOF));
      }
      else
	 clreol();
      linepos++;
      gotoxy(CS,linecount);
      pos=CS;
   }
   return;
}


unsigned long *movehome(unsigned long *startpos,unsigned long *lineno)
{
   fillwindow(startpos);  // home
   *lineno=1;
   textattr(BCOLOUR);
   gotoxy(67,1);
   cprintf("%06ld",*lineno);
   textattr(WCOLOUR);
   return(startpos);
}


unsigned long *moveend(unsigned long *finishpos, unsigned long *startpos,unsigned long *lineno, unsigned long totlines)
{
   unsigned long *aptr;

   aptr=finishpos-(RF-RS)-2;   //end
   if (aptr < startpos)
   {
      aptr=startpos;
      *lineno=1;
   }
   else
      *lineno=totlines-(RF-RS);
   fillwindow(aptr);
   textattr(BCOLOUR);
   gotoxy(67,1);
   cprintf("%06ld",*lineno);
   textattr(WCOLOUR);
   return(aptr);
}

void scrollleft(void)
{
   movetext(CS+1,RS,CF,RF,CS,RS);
   gotoxy(CF,RS);
   col++;
   gotoxy(57,1);
   textattr(BCOLOUR);
   cprintf("%03d",col);
   textattr(WCOLOUR);
return;
}

void scrollright(void)
{
   movetext(CS,RS,CF-1,RF,CS+1,RS);
   gotoxy(CS,RS);
   col--;
   gotoxy(57,1);
   textattr(BCOLOUR);
   cprintf("%03d",col);
   textattr(WCOLOUR);
return;
}



void display_line(void)
{
   unsigned char pos=CS;
   char c;
   int colcount,eol;

   eol=0;
   for (colcount=1;((colcount < col) && (!eol));colcount++)
   {
      c=fgetc(in);
      if ((c==10) || (c==EOF))
	 eol=1;
   }
   if (!eol)
   {
      do
      {
	 c=fgetc(in);
	 if (pos++ <= CF)
	 {
	    if ((c==10) || (c==EOF))
	       clreol();  // clear to end of line
	    else
	       putch(c);
	 }
      }  while ((c!=10) && (c!=EOF));
   }
   else
      clreol();
return;
}



void display_col(int pos,unsigned long *startline)
{
   int colcount,linecount=RS,eol;
   char c;          // col is global

   gotoxy(pos,RS);
   while (linecount++ <= RF)
   {
      fseek(in,*startline,SEEK_SET);
      eol=0;
      for (colcount=1;((colcount < col) && (!eol));colcount++)
      {
	 c=fgetc(in);
	 if ((c==10) || (c==EOF))
	    eol=1;
      }
      if (!eol)
      {
	 while ((colcount < pos+col) && (!eol))
	 {
	    c=fgetc(in);
	    if ((c==10) || (c==EOF))
	       eol=1;
	    colcount++;
	 }
	 if (!eol)
	    putch(c);
	 else
	    putch(' ');
      }
      else
	 putch(' ');
      gotoxy(pos,linecount);
      startline++;
   }
return;
}
