% QMW PhD Thesis Class
% 7.8.1998
% Modified from:

% File:         ssxdphil.cls
% Author:       J. K. Brook <joannab@cogs.susx.ac.uk>
% Date:         1997/03/10
% Purpose:      Sussex D.Phil. Thesis Class

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{qmwphd}
   [1997/07/22 QMW DCS PhD Thesis Class]

% based on 11pt report 

% default font Times-Roman

\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{stmaryrd}
\RequirePackage{times} % to get helvetica sans serif - berdine
\RequirePackage{mathptmx} 
\DeclareSymbolFont{symbols}{OMS}{cmsy}{m}{n} % to get sane \mathcal
                                              % letters - berdine

% Defaults to single-spacing throughout to save paper.
% The figure of 1.21 is based on the baselinestretch given in the
% doublespace package for one-and-a-half spacing at 11pt 
% (LaTeX Companion p. 53) 

% The spaced option only affects line spacing (and hence page breaks). 

\newcommand{\globalbaselinestretch}{1}

\DeclareOption{spaced}{%
   \AtBeginDocument{%
      \renewcommand{\globalbaselinestretch}{1.62}
      \renewcommand{\baselinestretch}{1.62}\normalsize
   }%
}

% The draft option centres the text (which works better with dvi viewers),
% prints `DRAFT' at the top of each page and the time and date at the
% bottom and suppresses page numbers (so that the draft thesis
% can be kept in a ring binder and sections replaced with longer or
% shorter versions).  It should not affect page breaks.

\newif\if@draft
\@draftfalse

\DeclareOption{draft}{%
   \@drafttrue%
%   \AtBeginDocument{%
%   \PassOptionsToClass{openright}{report}
%   }%
}

%-ae \DeclareOption{csrp}{%
%-ae    \PassOptionsToClass{twoside}{report}%
%-ae    \AtBeginDocument{%
%-ae       \renewcommand{\baselinestretch}{1}\normalsize
%-ae       \let\maketitle\makecover
%-ae    }%
%-ae }

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\LoadClass[11pt,a4paper]{report}

% defaults for customisable things 

\newcommand{\partnamestyle}{\LARGE\bfseries}
\newcommand{\chapnamestyle}{\LARGE\bfseries}
\newcommand{\secnamestyle}{\large\bfseries}
\newcommand{\subsecnamestyle}{\normalsize\bfseries}
\newcommand{\subsubsecnamestyle}{\normalsize\itshape}
\newcommand{\paranamestyle}{\normalsize\itshape}
\newcommand{\subparanamestyle}{\normalsize\itshape}

\newcommand{\abstractnamestyle}{\Large\bfseries}
\newcommand{\titlestyle}{\LARGE\bfseries}
\newcommand{\authornamestyle}{\large\bfseries}

\newcommand{\captionnamestyle}{\normalsize\itshape}
\newcommand{\captionstyle}{\normalsize}

\newcommand{\prefacename}{Preface}
\newcommand{\summaryname}{Abstract} % U. of London reg.s -ae
\newcommand{\acknowledgementsname}{Acknowledgements} 
%-ae \newcommand{\declarationname}{Declaration}
%-ae \newcommand{\signaturename}{Signature}


% ae begin.
\newcount\hour
\newcount\minute
\hour=\time
\divide\hour by 60
\minute=\time
\multiply\hour by 60
\advance\minute by -\hour
\divide\hour by 60

\def\timenow{\number\hour {.}%
\ifnum\minute<10 0\number\minute\else\number\minute\fi}

\def\today{\number\day\space
  \ifcase\month\or January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi\space
  \number\year}

\def\timetoday{\timenow, \today}
% ae end.


% A version of \raggedright that tries its best to balance the rhs of
% the paragraph.

\newcommand{\balancedraggedright}{%
   \raggedright
   \rightskip 0pt plus \textwidth
   \@rightskip \rightskip
   \parfillskip \rightskip
}

% Chapter-head displays for mainmatter/numbered chapters
% -- based on MIT Press Cognitive Science books

\renewcommand{\@makechapterhead}[1]{%
  \vspace*{1.32cm}%
  {\parindent\z@ \balancedraggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \chapnamestyle \@chapapp{} \chapnamestyle \thechapter
        \par\nobreak
        \vskip 1.2ex
    \fi
    \interlinepenalty\@M
    \chapnamestyle #1\par\nobreak
    \vskip 2.38cm 
    \rule{\linewidth}{0.2mm}%
    \vskip 15\p@
  }}

% for unnumbered and frontmatter sections e.g., preface

\renewcommand{\@makeschapterhead}[1]{%
  \vspace*{1.32cm}%
  {\parindent \z@ \balancedraggedright
    \normalfont
    \interlinepenalty\@M
    \chapnamestyle #1\par\nobreak
        \vskip 1.7ex
  }}

% Chapters run with a blank page before them if necessary to open right.

\renewcommand\part{%
   \clearpage
   \thispagestyle{headings}
   \if@openright{\cleardoublepage}\fi
   \thispagestyle{headings}
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \secdef\@part\@spart}


\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\vspace*{1.32cm}%
     \parindent\z@ \balancedraggedright \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \partnamestyle \partname~\thepart
       \par\nobreak
       \vskip 1.2ex
     \fi
     \interlinepenalty \@M
     \partnamestyle #2\par\nobreak}%
    \@endpart}
\def\@spart#1{%
    {\vspace*{1.32cm}%
     \parindent\z@ \balancedraggedright 
     \normalfont
     \interlinepenalty \@M
     \partnamestyle #1\par\nobreak}%
    \@endpart}
\def\@endpart{\vfil\newpage
              \if@twoside
                \null
                \thispagestyle{headings}%
                \newpage
              \fi
              \if@tempswa
                \twocolumn
              \fi}

\renewcommand\chapter{%
   \clearpage
   \thispagestyle{headings}
   \if@openright{\cleardoublepage}\fi
   \thispagestyle{headings}
   \global\@topnum\z@
   \@afterindentfalse
   \secdef\@chapter\@schapter
}

% Starred chapters update the running head.

\renewcommand\@schapter[1]{%
   \markboth{#1}{}%
   \if@twocolumn
      \@topnewpage[\@makeschapterhead{#1}]%
   \else
      \@makeschapterhead{#1}%
      \@afterheading
   \fi
}

% Regulations: Bound margin no less than 40mm (before binding), 
%              other margins no less than 20mm.

\topmargin -0.20in 
\textheight 24.3cm 
\textwidth  14.8cm  

% The figures below need an inch taken off them because LL decided to
% use DEK's (1in,1in) origin.  [Sigh]  The draft option centres the
% text for better viewing on xdvi.

\if@draft
   \oddsidemargin 3cm  \advance\oddsidemargin by -1in 
   \evensidemargin 3cm  \advance\evensidemargin by -1in
\else
   \oddsidemargin 4cm  \advance\oddsidemargin by -1in 
   \evensidemargin 2cm  \advance\evensidemargin by -1in 
\fi                                                    

% running heads

\renewcommand{\chaptermark}[1]{%
   \markboth{\@chapapp\ \thechapter.\enspace   #1}{}%
}
\renewcommand{\sectionmark}[1]{%
   \markright{\thesection.\enspace   #1}%
}

% For the one-sided version, we run with:
%
%                                           N.M. Section Title   Folio
%
% For the two-sided verso, we run with:
%
%    Folio   Chapter N. Chapter title
%
% For the two-sided recto, we run with:
%
%                                           N.M. Section Title   Folio
%

\if@draft
   \renewcommand{\ps@headings}{%
        \renewcommand{\@oddhead}{{\bf DRAFT}%
                                 \hfil\textit\rightmark}%
        \renewcommand{\@evenhead}{\textit\leftmark\hfil%
                                  {\bf DRAFT}}%
        \renewcommand{\@oddfoot}{\hfil\timetoday---\thepage}%
        \renewcommand{\@evenfoot}{\thepage---\timetoday\hfil}%
   }
\else
   \renewcommand{\ps@headings}{%
        \renewcommand{\@oddhead}{\hfil\textit\rightmark\quad\textrm\thepage}%
        \renewcommand{\@evenhead}{\textrm\thepage\quad\textit\leftmark\hfil}%
        \renewcommand{\@oddfoot}{}%
        \renewcommand{\@evenfoot}{\@oddfoot}%
   }
\fi

\pagestyle{headings}

% A number on every page. -ae

\if@draft
   \renewcommand{\ps@headings}{%
        \renewcommand{\@oddhead}{{\bf DRAFT}%
                                 \hfil}%
        \renewcommand{\@evenhead}{\hfil%
                                  {\bf DRAFT}}%
        \renewcommand{\@oddfoot}{\hfil\timetoday---\thepage}%
        \renewcommand{\@evenfoot}{\thepage---\timetoday\hfil}%
   }
\else
   \renewcommand{\ps@headings}{%
        \renewcommand{\@oddhead}{\hfil\thepage}%
        \renewcommand{\@evenhead}{\thepage\hfil}%
        \renewcommand{\@oddfoot}{}%
        \renewcommand{\@evenfoot}{\@oddfoot}%
   }
\fi

% All lists are set in single-spacing (this is a horrid hack!)
% Pablo prefers double spaced lists.
%\let\qmwphd@list=\list
%\renewcommand{\list}{%
%   \par
%   \renewcommand{\baselinestretch}{1}\@currsize
%   \qmwphd@list
%}

% hack parboxes to be single spaced
%? \makeatletter
\let\qmwphd@parbox=\parbox
\renewcommand{\parbox}{%
   \par
   \renewcommand{\baselinestretch}{1}\@currsize
   \qmwphd@parbox
}

% hack eqnarray too (was ...array@i - why?)
\let\qmwphd@eqnarray=\eqnarray
\renewcommand{\eqnarray}{%
   \par
   \renewcommand{\baselinestretch}{1}\@currsize
   \qmwphd@eqnarray
}

% footnotes are single-spaced and \small instead of \footnotesize.

\renewcommand{\@makefntext}[1]
{\renewcommand{\baselinestretch}{1}\small
\parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}

% This seems to work.

\newenvironment{notspaced}{\renewcommand{\baselinestretch}{1}\@currsize}{}

% sectioning commands adapted from report.cls
% reduced space measurements from Richard Dallaway's iot.sty

\renewcommand{\section}{\@startsection{section}{1}{\z@}
                                     {-3.5ex \@plus -1ex \@minus -.2ex}
                                     {1.5ex plus.2ex}
                                     {\balancedraggedright\secnamestyle}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     %{1.5ex \@plus .2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\balancedraggedright\subsecnamestyle}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
                                     %{-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {-1ex\@plus -0.5ex \@minus -.1ex}%
                                     %{1.5ex \@plus .2ex}%
                                     {0.25ex \@plus .1ex}%
                                     {\balancedraggedright\subsubsecnamestyle}}
\renewcommand{\paragraph}{\@startsection{paragraph}{4}{\z@}%
                                     %{3.25ex \@plus1ex \@minus.2ex}%
                                     {-1ex\@plus -0.5ex \@minus -.1ex}%
                                     {-1em}%
                                     {\paranamestyle}}
\renewcommand{\subparagraph}{\@startsection{subparagraph}{5}{\parindent}%
                                     %{3.25ex \@plus1ex \@minus .2ex}%
                                     {-1ex\@plus -0.5ex \@minus -.1ex}%
                                     {-1em}%
                                     {\subparanamestyle}}
% frontmatter

% order of frontmatter from regulations:

%Title page                    % U. of London says
%Summary                       % Abstract follows Title Page
%Contents                      % and then Contents etc.
%List of tables, figures, etc.

%Acknowledgements (if any) 

% mainmatter

%Chapters
%Bibliography

% backmatter

%Appendices

% title page definitions:
% \author, \title, \qualification, \university, \date (month and year of 
% submission)

\renewcommand\author[1]{\gdef\@author{#1}}
\renewcommand\title[1]{\gdef\@title{#1}}
\newcommand\qualification[1]{\gdef\@qualification{#1}}

\qualification{Doctor of Philosophy} 

% the university to which the degree is being submitted % = College -ae

\newcommand\university[1]{\gdef\@university{#1}}

\university{Queen Mary, University of London} % College name. -ae

%-ae \renewcommand{\today}{
%-ae \ifcase\month\or
%-ae   January\or February\or March\or April\or May\or June\or
%-ae   July\or August\or September\or October\or November\or December\fi,
%-ae   \space \number\year}
\newcommand{\yeartoday}{\number\year}

\renewcommand\date[1]{\gdef\@date{#1}}

%-ae % default to \today

%-ae \date{\today}
\date{\yeartoday}

% declaration

% No declaration. -ae
%-ae \newcommand{\declaration}[1]{\gdef\@declaration{#1}} 

% default from regulations 

%-ae \declaration{%
%-ae    I hereby declare that this thesis has not been submitted, either in
%-ae    the same or different form, to this or any other university for a
%-ae    degree.%
%-ae }

% Acknowledgements (optional).
\newenvironment{acknowledgements}{%
   \chapter*{\acknowledgementsname}%
   %\addcontentsline{toc}{chapter}{\acknowledgementsname}% 
}{%
   \clearpage
}

% preface -- sources and extent of any joint work

% No preface. -ae
%-ae \newenvironment{preface}{%
%-ae   \chapter*{\prefacename}%
%-ae  % \addcontentsline{toc}{chapter}{\prefacename}%
%-ae}{% 
%-ae   \clearpage
%-ae}

% summary (or abstract) -- 300 words 
% This shares a lot of code with \maketitle:

\newcommand{\ssxdphil@summary@maketitle@top}{{%
  \clearpage
  \thispagestyle{headings}% Was empty. -ae
  \balancedraggedright
  \vspace*{1.32cm} % same as chapter-head display
  \titlestyle\@title \par
  \bigskip
  \authornamestyle \@author \par
  \normalfont\normalsize
  \vspace{1cm}%
}}

\newcommand{\ssxdphil@summary@maketitle@bot}{{%
  \par\vfill\balancedraggedright
  Submitted for the degree of \@qualification \par
  \medskip
  \@university \par
  \medskip
  \@date \par
}}

% The summary is run single-spaced.

\newenvironment{summary}{%
  \cleardoublepage
  \ssxdphil@summary@maketitle@top
  \renewcommand{\baselinestretch}{1}\normalsize
  \section*{\summaryname}%
}{%
  \par
  \renewcommand{\baselinestretch}{\globalbaselinestretch}\normalsize
  \ssxdphil@summary@maketitle@bot
  \clearpage
}

% Title-page 

\renewcommand{\maketitle}{%
  \ssxdphil@summary@maketitle@top
  \ssxdphil@summary@maketitle@bot
  \clearpage
%-ae  \chapter*{\declarationname}%
%-ae  \@declaration\par
%-ae  \vspace{5cm}%
%-ae  \noindent\signaturename:\par
%-ae  \clearpage
}

% Gross document structure

%\newcommand{\frontmatter}{\clearpage\pagenumbering{roman}}
% All pages arabic. -ae 
\newcommand{\frontmatter}{\clearpage} 
%-ae \newcommand{\mainmatter}{\clearpage\pagenumbering{arabic}}
\newcommand{\mainmatter}{\clearpage}
\newcommand{\backmatter}{\clearpage\appendix}

% Float parameters

\setcounter{topnumber}{2}
\renewcommand{\topfraction}{.7}
\setcounter{bottomnumber}{1}
\renewcommand{\bottomfraction}{.3}
\setcounter{totalnumber}{3}
\renewcommand{\textfraction}{.2}
\renewcommand{\floatpagefraction}{.5}
\setcounter{dbltopnumber}{2}
\renewcommand{\dbltopfraction}{.7}
\renewcommand{\dblfloatpagefraction}{.5}

% Floats can't go at the bottom or in-line by default:

\renewcommand{\fps@figure}{tp}
\renewcommand{\fps@table}{tp}

% Caption name set in \captionnamestyle with name in \captionstyle.

%\renewcommand{\@makecaption}[2]{
% \medskip
% \setbox\@tempboxa\hbox{{\captionnamestyle#1.} {\captionstyle#2}}
% \ifdim \wd\@tempboxa >\hsize {\centering{\captionnamestyle #1.}
% {\captionstyle#2\par}} \else \hbox
%to\hsize{\hfil\box\@tempboxa\hfil}
% \fi}

% sje -- change caption style 
% This taken from report.cls
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

% Bibliography adds a toc entry.

\renewenvironment{thebibliography}[1]
     {\chapter*{\bibname}%
        \addcontentsline{toc}{chapter}{\bibname}%%%% ADDITION
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
%-ae            \if@openbib
%-ae              \advance\leftmargin\bibindent
%-ae              \itemindent -\bibindent
%-ae              \listparindent \itemindent
%-ae              \parsep \z@
%-ae            \fi
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
%-ae      \if@openbib
%-ae        \renewcommand\newblock{\par}%
%-ae      \else
        \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
%-ae      \fi
      \sloppy\clubpenalty4000\widowpenalty4000%
      \sfcode`\.=\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

% Index adds a toc entry.

\renewenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
                \markboth{\indexname}{}%
                \addcontentsline{toc}{chapter}{\indexname}%%%% ADDITION
                \thispagestyle{headings}\parindent\z@%%% Was empty. -ae
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}

%%% CHANGELOG

% jkb,  1995/09/19: Created MIT Press-style chapterhead

% jkb,  1995/09/19: Sorted out top/bottom margins 

% jkb,  1995/09/19: Reduced space around headings

% jkb,  1995/09/20: Roughed out titlepage with ssxcrest

% jkb,  1995/09/20: Roughed out declaration page

% jkb,  1995/09/20: Changed epsfig to graphics

% jkb,  1995/09/20: Changed font package from times to mathptm --
%    has the effect of changing \sfdefault from phv to cmss
%    and \ttdefault from Courier 

% jkb,  1995/09/20: Discovered that thesis.sty has bottom-centred
%    page-numbers and is still approved by the PG office

% jkb,  1995/09/20: Added book.cls-style \frontmatter and \mainmatter commands

% jkb,  1995/10/20: Tried creating left-justified title-page and summary

% jkb,  1995/10/20: Added default empty acknowledgements command

% jkb,  1995/10/25: First draft version completed

% ASAJ, 1995/12/07: Made summary, etc. non-declarative.

% ASAJ, 1995/12/07: Removed the \summary* commands.

% ASAJ, 1995/12/07: Removed \makefrontmatter.

% ASAJ, 1995/12/07: Replaced `spacing' option by `thesis' option, and
%    made it one-sided by default.  Added the csrp option.

% ASAJ, 1995/12/07: Added \backmatter.

% ASAJ, 1995/12/07: Added a ghastly hack to get all lists set
%    single-spaced. 

% ASAJ, 1995/12/07: Made title page and summary page similar in style.

% ASAJ, 1995/12/08: Made the title page and summary page share code,
%    and removed the crest.

% ASAJ, 1995/12/08: Redesigned the running heads.

% ASAJ, 1995/12/08: Made section heads ragged right.

% ASAJ, 1995/12/08: Made ragged right material try to be more balanced.

% ASAJ, 1995/12/08: Added \clearpage before setting the pagestyle.

% jkb,  1995/12/13: Removed \addtoclines from acknowledgements and preface
%    to sort out conflict between lack of folios and contents-page numbering

% jkb,  1995/12/13: Fixed margins in twoside version -- both were
%    set using \oddsidemargin.  For CSRP version want same
%    textwidth so as not to cause problems with explicitly sized
%    figures etc., but can have more reasonable margins.

% jkb,  1996/02/12: Added \university command.

% jkb,  1996/02/12: Added final full point to D. Phil. (Hart's Rules).

% jkb,  1996/02/14: Tidied comments. First release version completed.

% ASAJ, 1996/04/01: Made the default title size \LARGE rather than
%    \Large, so it's the same size as chapter openings.

% ASAJ, 1996/05/03: Replaced \chaptername by \@chapapp in running heads.

% ASAJ, 1996/05/14: \chapter* now updates the running head.  \theindex
%    and \thebibliography now add to the toc.

% ASAJ, 1996/09/20: Footnotes are now in \footnotesize.  `Submitted
%    for the degree of' stuff is now at the page foot.

% sje, 1997/02/05: Better setting of captions

% ASAJ 1997/03/10: Corrected setting of running heads for index.


