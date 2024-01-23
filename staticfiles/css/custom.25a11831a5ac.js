document.addEventListener("DOMContentLoaded", function () {
  const nav = document.getElementById("main__nav");
  const hamburger = document.getElementById("hamburger");

  nav.addEventListener("click", handleNav);
  hamburger.addEventListener("click", handleNav);

  function handleNav() {
    if (!nav.classList.contains("main__nav--active")) {
      nav.classList.add("main__nav--active");
      hamburger.classList.add("hamburger--active");
    } else {
      nav.classList.remove("main__nav--active");
      hamburger.classList.remove("hamburger--active");
    }
  }
});

document.addEventListener("DOMContentLoaded", function () {
  // Variables
  const tabLinks = document.querySelectorAll("#tabs-section .tab-link");
  const tabBodies = document.querySelectorAll("#tabs-section .tab-body");
  let timerOpacity;

  // Toggle Class
  const init = () => {
    // Select the input box element

    const inputBox = document.getElementById("chatBoxInput");

    // Add an event listener for keydown event
    inputBox.addEventListener("keydown", function (event) {
      // Check if only Enter key is pressed
      if (event.key === "Enter" && !event.shiftKey) {
        // Prevent the default action to stop going to the next line
        event.preventDefault();
      }
      // Check if Shift and Enter keys are pressed together
      else if (event.shiftKey && event.key === "Enter") {
        // Prevent the default action to avoid form submission
        event.preventDefault();
        // Add a new line at the current cursor position
        const cursorPos = inputBox.selectionStart;
        const textBeforeCursor = inputBox.value.substring(0, cursorPos);
        const textAfterCursor = inputBox.value.substring(cursorPos);
        inputBox.value = textBeforeCursor + "\n" + textAfterCursor;
        // Move the cursor to the next line
        inputBox.selectionStart = cursorPos + 1;
        inputBox.selectionEnd = cursorPos + 1;
      }
    });

    // Menu Click
    tabLinks.forEach((link) => {
      link.addEventListener("click", function (e) {
        // Prevent Default
        e.preventDefault();
        e.stopPropagation();

        // Clear Timers
        window.clearTimeout(timerOpacity);

        // Toggle Class Logic
        // Remove Active Classes
        tabLinks.forEach((tabLink) => tabLink.classList.remove("active"));
        tabBodies.forEach((tabBody) => {
          tabBody.classList.remove("active", "active-content");
        });

        // Add Active Classes
        this.classList.add("active");
        document
          .querySelector(this.getAttribute("href"))
          .classList.add("active");

        // Opacity Transition Class
        timerOpacity = setTimeout(() => {
          document
            .querySelector(this.getAttribute("href"))
            .classList.add("active-content");
        }, 50);
      });
    });

    var element = $(".floating-chat");
    var myStorage = localStorage;

    if (!myStorage.getItem("chatID")) {
      myStorage.setItem("chatID", createUUID());
    }

    setTimeout(function () {
      element.addClass("enter");
    }, 1000);

    element.click(openElement);

    function openElement() {
      var messages = element.find(".floating-chat-messages");
      var textInput = element.find(".text-box");
      element.find(">i").hide();
      element.addClass("expand");
      element.find(".chat").addClass("enter");
      var strLength = textInput.val().length * 2;
      textInput.keydown(onMetaAndEnter).prop("disabled", false).focus();
      element.off("click", openElement);
      element.find(".header button").click(closeElement);
      element.find("#sendMessage").click(sendNewMessage);
      messages.scrollTop(messages.prop("scrollHeight"));
    }

    function closeElement() {
      element.find(".chat").removeClass("enter").hide();
      element.find(">i").show();
      element.removeClass("expand");
      element.find(".header button").off("click", closeElement);
      element.find("#sendMessage").off("click", sendNewMessage);
      element
        .find(".text-box")
        .off("keydown", onMetaAndEnter)
        .prop("disabled", true)
        .blur();
      setTimeout(function () {
        element.find(".chat").removeClass("enter").show();
        element.click(openElement);
      }, 500);
    }

    function createUUID() {
      // http://www.ietf.org/rfc/rfc4122.txt
      var s = [];
      var hexDigits = "0123456789abcdef";
      for (var i = 0; i < 36; i++) {
        s[i] = hexDigits.substr(Math.floor(Math.random() * 0x10), 1);
      }
      s[14] = "4"; // bits 12-15 of the time_hi_and_version field to 0010
      s[19] = hexDigits.substr((s[19] & 0x3) | 0x8, 1); // bits 6-7 of the clock_seq_hi_and_reserved to 01
      s[8] = s[13] = s[18] = s[23] = "-";

      var uuid = s.join("");
      return uuid;
    }

    function sendNewMessage() {
      var userInput = $(".text-box");
      var newMessage = userInput
        .html()
        .replace(/\<div\>|\<br.*?\>/gi, "\n")
        .replace(/\<\/div\>/g, "")
        .trim()
        .replace(/\n/g, "<br>");

      if (!newMessage) return;

      var messagesContainer = $(".floating-chat-messages");

      messagesContainer.append(
        ['<li class="self">', newMessage, "</li>"].join("")
      );

      // clean out old message
      userInput.html("");
      // focus on input
      userInput.focus();

      messagesContainer.finish().animate(
        {
          scrollTop: messagesContainer.prop("scrollHeight"),
        },
        250
      );
    }

    function onMetaAndEnter(event) {
      if ((event.metaKey || event.ctrlKey) && event.keyCode == 13) {
        sendNewMessage();
      }
    }
  };

  // Document Ready
  init();
});
