// Based on a script by Kathie Decora : katydecorah.com/code/lunr-and-jekyll/

// Create the lunr index for the search
var index = elasticlunr(function () {
  this.addField('title')
  this.addField('author')
  this.addField('layout')
  this.addField('content')
  this.setRef('id')
});

// Add to this index the proper metadata from the Jekyll content

index.addDoc({
  title: "My Favorite Films",
  author: "Yotam Perlitz",
  layout: "post",
  content: "This is an ongoing list of no particular order containing some of favorite films, all of the movies in the list are such that I consider either intresting, importent of just fun.\n\nI have choosen to add Rotten-Tomatoes top-critic score1 which I find to be the most informative of the scores2. Enjoy the list :)\n\n\n  The Square (2017) (77%)\n\n\nA Ruben Östlund film about modern art, modern civilization and how they went wrong.\n\n\n  FORCE MAJEURE (2015) (94%)\n\n\nA Ruben Östlund film about an unexpected turn of events\n\n\n  Elle (2016) (85%)\n\n\nA Paul Verhoeven film featuring an amazing performance by  Isabelle Huppert about a strong person handeling a weak spot.\n\n\n  Things to come. (100%)\n\n\nA Mia Hansen-Løve film featuring Isabelle Huppert about the decision we make in life.\n\n\n  There will be blood (94%)\n\n\nA Paul Thomas Anderson film featuring Daniel Day-Lewis at his best about a miner during the black gold rush. The film is strong and troubling, combined with one of the best soundtracks I heard by Radiohead’s Jonny Greenwood, it is a must.\n\n\n  SPIDER-MAN: INTO THE SPIDER-VERSE (95%)\n\n\nA Bob Persichetti, Peter Ramsey, Rodney Rothman Film\n\n\n  \n    \n      Rotten Tommatoes top critics score is a score made by averaging over ~30 professional critics from major newspapers, each critic gives a yey or ney, if for example a film gets a score of 80%, it means that 8 out of every 10 top critics would recommend watching the film. &#x21a9;&#xfe0e;\n    \n    \n      Another good indication which I learned from Shira is looking at the reviews by the NY and LA times, I find that I can trust their opinion. &#x21a9;&#xfe0e;\n    \n  \n\n",
  id: 0
});
console.log( jQuery.type(index) );

// Builds reference data (maybe not necessary for us, to check)
var store = [{
  "title": "My Favorite Films",
  "author": "Yotam Perlitz",
  "layout": "post",
  "link": "/texts/My%20favorite%20films/",
}
]

// Query
var qd = {}; // Gets values from the URL
location.search.substr(1).split("&").forEach(function(item) {
    var s = item.split("="),
        k = s[0],
        v = s[1] && decodeURIComponent(s[1]);
    (k in qd) ? qd[k].push(v) : qd[k] = [v]
});

function doSearch() {
  var resultdiv = $('#results');
  var query = $('input#search').val();

  // The search is then launched on the index built with Lunr
  var result = index.search(query);
  resultdiv.empty();
  if (result.length == 0) {
    resultdiv.append('<p class="">No results found.</p>');
  } else if (result.length == 1) {
    resultdiv.append('<p class="">Found '+result.length+' result</p>');
  } else {
    resultdiv.append('<p class="">Found '+result.length+' results</p>');
  }
  // Loop through, match, and add results
  for (var item in result) {
    var ref = result[item].ref;
    var searchitem = '<div class="result"><p><a href="'+store[ref].link+'?q='+query+'">'+store[ref].title+'</a></p></div>';
    resultdiv.append(searchitem);
  }
}

$(document).ready(function() {
  if (qd.q) {
    $('input#search').val(qd.q[0]);
    doSearch();
  }
  $('input#search').on('keyup', doSearch);
});
