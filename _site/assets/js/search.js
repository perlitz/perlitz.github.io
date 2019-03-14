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
  content: "This is an ongoing list of my favorite films, all of the movies in the list are such that ?I consider either intresting, importent of just fun, I have choosen to add Rotten-Tomatoes top-critic score which I find to be the most informative of the scores. Enjoy the list :)\n\n\n  The Square (2017) (rt-tc : 77%)\n\n\nA Ruben Östlund film about modern art, modern civilization and how they went wrong.\n\n\n  FORCE MAJEURE (2015) (rt-tc : 94%)\n\n\nA Ruben Östlund film about an unexpected turn of events\n",
  id: 0
});
console.log( jQuery.type(index) );

// Builds reference data (maybe not necessary for us, to check)
var store = [{
  "title": "My Favorite Films",
  "author": "Yotam Perlitz",
  "layout": "post",
  "link": "/texts/another-one/",
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
