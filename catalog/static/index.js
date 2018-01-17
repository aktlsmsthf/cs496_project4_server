(function()
{
	var canvas = document.querySelector( "#canvas" );
	var canvasObj = document.getElementById("canvas");


	var context = canvas.getContext( "2d" );
	canvas.width = 560;
	canvas.height = 560;

	var Mouse = { x: 0, y: 0 };
	var lastMouse = { x: 0, y: 0 };
	context.fillStyle="white";
	context.fillRect(0,0,canvas.width,canvas.height);
	context.color = "black";
    context.lineJoin = context.lineCap = 'round';
	// var flag = 0;

	debug();

	canvas.addEventListener( "mousemove", function( e )
	{
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

		Mouse.x = e.pageX - this.offsetLeft;
		Mouse.y = e.pageY - this.offsetTop;
		flag = 1;
	}, false );

	canvas.addEventListener( "mousedown", function( e )
	{
		canvas.addEventListener( "mousemove", onPaint, false );

	}, false );

	canvas.addEventListener( "contextmenu", function( e )
	{


		var img = canvasObj.toDataURL();
		img = encodeURIComponent(img);
		var label = $("#numberinput").val()

		$.ajax({
			type: "POST",
			url: '/catalog/test/',
			data: label+"label"+img,
			success: function(data){
				context.clearRect( 0, 0, 560, 560 );
				context.fillStyle="white";
				context.fillRect(0,0,canvas.width,canvas.height);
				flag = 0;

			}
		});

		//window.event.returnValue = false;
		


		//$('#result').replaceWith('<div id = "result"> <p></p></div>');

	},false );

	canvas.addEventListener( "mouseup", function()
	{
		/*if(flag !=0){
			var img = canvasObj.toDataURL();
			img = encodeURIComponent(img)
			$.ajax({
				type: "POST",
				url: '/catalog/test/',
				data: img,
				success: function(data){
					$('#result').replaceWith('<div id = "result">' + data+ '</div>');
				}
			});
		}*/
		canvas.removeEventListener( "mousemove", onPaint, false );

	}, false );



	var onPaint = function()
	{
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

		context.beginPath();
		context.moveTo( lastMouse.x, lastMouse.y );
		context.lineTo( Mouse.x, Mouse.y );
		context.closePath();
		context.stroke();
	};

	function debug()
	{
		/* CLEAR BUTTON */
		var clearButton = $( "#clearButton" );

		clearButton.on( "click", function()
		{

				context.clearRect( 0, 0, 560, 560 );
				context.fillStyle="white";
				context.fillRect(0,0,canvas.width,canvas.height);

		});

		/* COLOR SELECTOR */

		$( "#colors" ).change(function()
		{
			var color = $( "#colors" ).val();
			context.color = color;
		});

		/* LINE WIDTH */

		$( "#lineWidth" ).change(function()
		{
			context.lineWidth = $( this ).val();
		});
	}
}());
