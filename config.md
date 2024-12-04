<!--
Add here global page variables to use throughout your website.
-->

+++
author = "Shiyi S"
mintoclevel = 3

# uncomment and adjust the following line if the expected base URL of your website is something like [www.thebase.com/yourproject/]

# please do read the docs on deployment to avoid common issues: https://franklinjl.org/workflow/deploy/#deploying_your_website

# prepath = "yourproject"

# Add here files or directories that should be ignored by Franklin, otherwise

# these files might be copied and, if markdown, processed by Franklin which

# you might not want. Indicate directories by ending the name with a `/`.

# Base files such as LICENSE.md and README.md are ignored by default.

ignore = ["node_modules/"]

# RSS (the website_ must be defined to get RSS)

generate_rss = true
website_title = "Franklin Template"
website_descr = "Example website using Franklin"
website_url   = "https://tlienart.github.io/FranklinTemplates.jl/"
+++

@def title = "NLPwShiyi | Natural Language Processing with Shiyi"
@def website_description = "Website for documenting my journey of understanding Natural Language Processing"
@def prepath = "nlpwme"


@def lang = "julia"
@def author = "Shiyi S"

<!-- HEADER SPECS
  NOTE:
  - use_header_img:     to use an image as background for the header
  - header_img_path:    either a path to an asset or a SVG like here. Note that
                        the path must be CSS-compatible.
  - header_img_style:   additional styling, for instance whether to repeat
                        or not. For a SVG pattern, use repeat, otherwise use
                        no-repeat.
  - header_margin_top:  vertical margin above the header, if <= 55px there will
                        be no white space, if >= 60 px, there will be white
                        space between the navbar and the header. (Ideally
                        don't pick a value between the two as the exact
                        look is browser dependent). When use_hero = true,
                        hero_margin_top is used instead.

  - use_hero:           if false, main bar stretches from left to right
                        otherwise boxed
  - hero_width:         width of the hero, for instance 80% will mean the
                        hero will stretch over 80% of the width of the page.
  - hero_margin_top     used instead of header_margin_top if use_hero is true

  - add_github_view:    whether to add a "View on GitHub" button in header
  - add_github_star:    whether to add a "Star this package" button in header
  - github_repo:        path to the GitHub repo for the GitHub button
-->

@def use_header_img     = false
@def use_hero           = true
@def hero_width         = "80%"
@def hero_margin_top    = "100px"

@def add_github_view  = true
@def add_github_star  = true
@def github_repo      = "https://github.com/shiyisteezin/nlpwme"

<!-- SECTION LAYOUT
NOTE:
  - section_width:  integer number to control the default width of sections
                    you can also set it for individual sections by specifying
                    the width argument: `\begin{:section, ..., width=10}`.
-->

@def section_width = 10

<!-- COLOR PALETTE
You can use Hex, RGB or SVG color names; these tools are useful to choose:
  - color wheel: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Colors/Color_picker_tool
  - color names: https://developer.mozilla.org/en-US/docs/Web/CSS/color_value

NOTE:
  - header_color:      background color of the header
  - link_color:        color of links
  - link_hover_color:  color of links when hovered
  - section_bg_color:  background color of "secondary" sections to help
                       visually separate between sections.
  - footer_link_color: color of links in the footer
-->

@def header_color       = "#3f6388"
@def link_color         = "#2669DD"
@def link_hover_color   = "teal"
@def section_bg_color   = "#f6f8fa"
@def footer_link_color  = "cornflowerblue"

<!-- CODE LAYOUT
NOTE:
  - highlight_theme:    theme for the code, pick one from
                        https://highlightjs.org/static/demo/ for instance
                        "github" or "atom-one-dark"; use lower case and replace
                        spaces with `-`.
  - code_border_radius: how rounded the corners of code blocks should be
  - code_output_indent: how much left-identation to add for "output blocks"
                        (results of the evaluation of code blocks), use 0 if
                        you don't want indentation.
-->

@def highlight_theme    = "atom-one-dark"
@def code_border_radius = "10px"
@def code_output_indent = "15px"

<!-- YOUR DEFINITIONS
See franklinjl.org for more information on how to introduce your own
definitions and how they can be useful.
-->

\newcommand{\note}[1]{@@note @@title 💡 @@#1@@}
\newcommand{\warn}[1]{@@warning @@title 💡 Warning!@@ @@content #1 @@ @@}

\newcommand{\E}{\mathbb E}
\newcommand{\R}{\mathbb R}
\newcommand{\P}{\mathbb P}
\newcommand{\N}{\mathbb N}
\newcommand{\Sc}{\mathcal S}
\newcommand{\bx}{{\bf x}}
\newcommand{\by}{{\bf y}}
\newcommand{\be}{{\bf e}}
\newcommand{\ba}{{\bf a}}
\newcommand{\bb}{{\bf b}}
\newcommand{\bv}{{\bf v}}
\newcommand{\bw}{{\bf w}}

<!--
Add here global latex commands to use throughout your pages.
-->

\newcommand{\scal}[1]{\langle #1 \rangle}

<!-- INTERNAL DEFINITIONS =====================================================
===============================================================================
These definitions are important for the good functioning of some of the
commands that are defined and used in PkgPage.jl
-->

@def sections        = Pair{String,String}[]
@def section_counter = 1
@def showall         = true

@def mintoclevel = 2

<!--
Add here files or directories that should be ignored by Franklin, otherwise
these files might be copied and, if markdown, processed by Franklin which
you might not want. Indicate directories by ending the name with a `/`.
-->

@def ignore = ["node_modules/", "franklin", "franklin.pub"]

<!--
Add here global latex commands to use throughout your
pages. It can be math commands but does not need to be.
For instance:
* \newcommand{\phrase}{This is a long phrase to copy.}
-->

\newcommand{\R}{\mathbb R}
\newcommand{\E}{\mathbb E}
\newcommand{\P}{\mathbb P}
\newcommand{\N}{\mathbb N}

\newcommand{\scal}[1]{\langle #1 \rangle}
<!-- \newcommand{\blurb}[1]{~~~<p style="font-size: 1.05em; color: #333; line-height:1.5em"> #1 </p>~~~} -->

<!-- \newcommand{\youtube}[1]{~~~<iframe width="1020" height="574" src="https://www.youtube.com/embed/~~~#1~~~" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>~~~} -->

\newcommand{\card}[2]{
  @@card
    @@container
      ~~~
      <h2> #1 </h2>
      ~~~
      @@content #2 @@
    @@
  @@
}
