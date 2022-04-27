---
title: "Appen tricks"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- python
star: false
category: blog
author: yotam
description:  Tricks for writing experiments with Appen
---

# Appen tricks

### Variable number of questions

```
{% if {{column_name}} != "None" and {{column_name}} != "" %}
Whatever you want to do with column_name
{% endif %}
```

### Add bold

Add this to the javascript:

```
require(['jquery-noconflict', 'bootstrap-modal', 'bootstrap-tooltip', 'bootstrap-popover', 'jquery-cookie'], function ($) {
  Window.implement('$', function (el, nc) {
    return document.id(el, nc, this.document);
  });
  var $ = window.jQuery;
  $(document).ready(function () {
    console.log('test')
    $(".missed").replaceWith("<div class='alert alert-info alert-block' style='line-height:1.8;font-size:1.4em'><h3><strong>Hmmmm...</strong></h3>Looks like you got a test question incorrect.<br/>Please review the instructions again.</div>");
    $("#review-instructions").addClass("collapse in");
  });
});
```

Then, bold with this for example for the work `number`:

```
<span style="color: #1a5d8b"><strong>number</strong></span>
```

